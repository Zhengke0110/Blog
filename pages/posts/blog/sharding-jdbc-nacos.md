---
title: 基于 SPI 机制修改 ShardingJDBC 底层，实现 Nacos 配置数据源
date: 2025-03-04
draft: true
lang: zh
---

# 背景介绍

在微服务架构中，配置管理是一个重要的问题。特别是对于数据库分片等复杂配置，如何实现配置的动态更新和统一管理变得尤为关键。ShardingJDBC 是一款优秀的分库分表中间件，但其默认的配置方式主要依赖于本地文件，这在分布式环境下管理起来较为困难。

Nacos 作为阿里巴巴开源的配置中心和服务发现中心，能够很好地解决配置管理的问题。将 ShardingJDBC 的配置迁移到 Nacos 中，可以实现配置的集中管理和动态更新。

本文将介绍如何基于 SPI 机制修改 ShardingJDBC 底层，使其能够从 Nacos 获取配置，从而实现更灵活的数据源配置管理。

## 技术原理

ShardingJDBC 提供了 SPI (Service Provider Interface) 机制，允许开发者通过实现特定接口来扩展其功能。在本例中，我们将实现 `ShardingSphereDriverURLProvider` 接口，使 ShardingJDBC 能够从 Nacos 获取配置。

整体流程如下：
1. 应用启动时，通过 JDBC URL 指向 Nacos 上的配置文件
2. 自定义的 `NacosDriverURLProvider` 解析 URL，连接 Nacos 服务器
3. 从 Nacos 获取配置内容并解析为 ShardingJDBC 可识别的格式
4. ShardingJDBC 根据配置初始化数据源

# 第一步: 配置 pom.xml 文件

在数据源模块的 pom.xml 文件中添加以下依赖。我们需要 ShardingJDBC 核心包和 Nacos 客户端来实现功能。

```xml
<dependency>
    <groupId>org.apache.shardingsphere</groupId>
    <artifactId>shardingsphere-jdbc-core</artifactId>
    <version>5.3.2</version>
</dependency>
<dependency>
    <groupId>com.alibaba.nacos</groupId>
    <artifactId>nacos-client</artifactId>
    <!-- 建议使用 2.1.0 或更高版本 -->
    <version>2.1.0</version>
</dependency>
<!-- 如果使用 HikariCP 连接池，需要添加相应依赖 -->
<dependency>
    <groupId>com.zaxxer</groupId>
    <artifactId>HikariCP</artifactId>
    <version>4.0.3</version>
</dependency>
```

# 第二步: 配置 Nacos 配置文件

我们需要在 Nacos 上创建两个配置文件：

## live-user-shardingjdbc.yaml 文件 用于配置数据源

这个文件定义了具体的分片规则、数据源配置等信息。

```yaml
dataSources:
  user_master: ##新表，重建的分表
    dataSourceClassName: com.zaxxer.hikari.HikariDataSource
    driver-class-name: com.mysql.cj.jdbc.Driver
    jdbcUrl: jdbc:mysql://localhost:8808/live_user?useUnicode=true&characterEncoding=utf8
    username: root
    password: root

  user_slave0: ##新表，重建的分表
    dataSourceClassName: com.zaxxer.hikari.HikariDataSource
    driver-class-name: com.mysql.cj.jdbc.Driver
    jdbcUrl: jdbc:mysql://localhost:8808/live_user?useUnicode=true&characterEncoding=utf8
    username: root
    password: root
rules:
  - !READWRITE_SPLITTING
    dataSources:
      user_ds:
        staticStrategy:
          writeDataSourceName: user_master
          readDataSourceNames:
            - user_slave0 # 可配置多个
  - !SINGLE
    defaultDataSource: user_ds # 不分表分分库的默认数据源
  - !SHARDING
    tables:
      t_user:
        actualDataNodes: user_ds.t_user_${(0..99).collect(){it.toString().padLeft(2,'0')}}
        tableStrategy:
          standard:
            shardingColumn: user_id
            shardingAlgorithmName: t_user-inline
      t_user_tag:
        actualDataNodes: user_ds.t_user_tag_${(0..99).collect(){ it.toString().padLeft(2,'0') } }
        tableStrategy:
          standard:
            shardingColumn: user_id
            shardingAlgorithmName: t_user_tag-inline
    shardingAlgorithms:
      t_user-inline:
        type: INLINE
        props:
          algorithm-expression: t_user_${(user_id % 100).toString().padLeft(2,'0')}
      t_user_tag-inline:
        type: INLINE
        props:
          algorithm-expression: t_user_tag_${(user_id % 100).toString().padLeft(2,'0')}
props:
  sql-show: true
```

上面的配置文件定义了：

1. **两个数据源**：`user_master`（主库）和 `user_slave0`（从库）
2. **读写分离规则**：写操作使用主库，读操作使用从库
3. **分片规则**：
   - 对 `t_user` 表按 `user_id` 进行分表，共分 100 张表
   - 对 `t_user_tag` 表按 `user_id` 进行分表，共分 100 张表
4. **分片算法**：使用 INLINE 类型，按 `user_id % 100` 的规则进行分表

## live-user-provider.yaml 用于项目的启动配置

```yaml
spring:
  application:
    name: live-user-provider
  datasource:
    driver-class-name: org.apache.shardingsphere.driver.ShardingSphereDriver
    # url: jdbc:shardingsphere:classpath:db-sharding.yaml  # 传统方式使用本地配置文件
    url: jdbc:shardingsphere:nacos:127.0.0.1:8848:live-user-shardingjdbc.yaml  # 使用Nacos配置
    hikari:
      pool-name: user-pool
      minimum-idle: 15
      maximum-pool-size: 300
      connection-timeout: 4000
      max-lifetime: 60000
  data:
    redis:
      port: 6379
      host: 127.0.0.1
      lettuce:
        pool:
          min-idle: 10
          max-active: 50
          max-idle: 20
dubbo:
  application:
    name: live-user-application
  registry:
    address: nacos://127.0.0.1:8848
  server: true
  protocol:
    name: dubbo
    port: 9090

rocketmq:
  producer:
    name-srv: 127.0.0.1:9876
    send-time-out: 3000
    retry-times: 3
    group-name: ${spring.application.name}
  consumer:
    name-srv: 127.0.0.1:9876
    group-name: ${spring.application.name}
```

关键配置说明：
- `driver-class-name`: 使用 ShardingSphereDriver 作为驱动
- `url`: 使用我们自定义的 Nacos URL 格式，指向 Nacos 上的配置文件
  - 格式为：`jdbc:shardingsphere:nacos:<Nacos服务地址>:<Nacos配置文件名>`

# 第三步: 编写 Nacos 配置类, 实现远程获取功能(支持鉴权)

这是本文的核心部分，我们需要实现 `ShardingSphereDriverURLProvider` 接口，使 ShardingJDBC 能够从 Nacos 获取配置。

```java
import com.alibaba.nacos.api.NacosFactory;
import com.alibaba.nacos.api.PropertyKeyConst;
import com.alibaba.nacos.api.config.ConfigService;
import com.alibaba.nacos.api.exception.NacosException;
import org.apache.shardingsphere.driver.jdbc.core.driver.ShardingSphereDriverURLProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StringUtils;

import java.util.Properties;

/**
 * 实现 ShardingSphereDriverURLProvider 接口，支持从 Nacos 获取配置
 */
public class NacosDriverURLProvider implements ShardingSphereDriverURLProvider {
    private static Logger logger = LoggerFactory.getLogger(NacosDriverURLProvider.class);
    private static final String NACOS_TYPE = "nacos:";
    private static final String GROUP = "DEFAULT_GROUP";  // Nacos 默认分组

    /**
     * 判断当前 Provider 是否接受此 URL
     */
    @Override
    public boolean accept(String url) {
        return url != null && url.contains(NACOS_TYPE);
    }

    /**
     * 获取配置内容
     * @param url JDBC URL，格式为：jdbc:shardingsphere:nacos:<Nacos地址>:<配置文件名>[?username=xxx&&password=yyy&&namespace=zzz]
     * @return 配置文件内容的字节数组
     */
    @Override
    public byte[] getContent(final String url) {
        if (StringUtils.isEmpty(url)) {
            return null;
        }

        // 提取 nacos URL 部分: 如 127.0.0.1:8848:live-user-shardingjdbc.yaml
        String nacosUrl = url.substring(url.lastIndexOf(NACOS_TYPE) + NACOS_TYPE.length());
        String[] nacosStr = nacosUrl.split(":");
        String serverAddr = nacosStr[0] + ":" + nacosStr[1];  // Nacos 服务器地址
        String nacosFileStr = nacosStr[2];  // 配置文件名及可能的参数

        // 设置 Nacos 客户端属性
        Properties properties = new Properties();
        properties.setProperty(PropertyKeyConst.SERVER_ADDR, serverAddr);

        // 检查是否提供了额外参数（用户名、密码、命名空间等）
        String dataId;
        if (nacosFileStr.contains("?")) {
            String[] nacosFileProp = nacosFileStr.split("\\?");
            dataId = nacosFileProp[0];  // 配置文件名

            // 处理额外参数
            if (nacosFileProp.length > 1) {
                String[] acceptProp = nacosFileProp[1].split("&&");
                for (String propertyName : acceptProp) {
                    String[] propertyItem = propertyName.split("=");
                    if (propertyItem.length == 2) {
                        String key = propertyItem[0];
                        String value = propertyItem[1];

                        // 处理认证相关属性
                        switch (key) {
                            case "username":
                                properties.setProperty(PropertyKeyConst.USERNAME, value);
                                break;
                            case "password":
                                properties.setProperty(PropertyKeyConst.PASSWORD, value);
                                break;
                            case "namespace":
                                properties.setProperty(PropertyKeyConst.NAMESPACE, value);
                                break;
                        }
                    }
                }
            }
        } else {
            // 不包含查询参数的简单情况
            dataId = nacosFileStr;
        }

        try {
            // 创建 Nacos 配置服务实例
            ConfigService configService = NacosFactory.createConfigService(properties);
            // 获取配置内容，超时时间设为 6 秒
            String content = configService.getConfig(dataId, GROUP, 6000);
            logger.info("从 Nacos 获取配置内容成功，dataId: {}", dataId);
            return content.getBytes();
        } catch (NacosException e) {
            logger.error("从 Nacos 获取配置失败", e);
            throw new RuntimeException("从 Nacos 获取配置失败", e);
        }
    }
}
```

## SPI 配置文件

为了让 ShardingJDBC 能够发现我们实现的 Provider，需要创建 SPI 配置文件：

```
src/main/resources/META-INF/services/org.apache.shardingsphere.driver.jdbc.core.driver.ShardingSphereDriverURLProvider
```

文件内容为实现类的全限定名：

```
com.your.package.NacosDriverURLProvider
```

# 第四步: 创建 SPI 配置文件

在项目的 resources 目录下，创建目录 `META-INF/services`，并添加一个文件，文件名为接口的全限定名：

```
org.apache.shardingsphere.driver.jdbc.core.driver.ShardingSphereDriverURLProvider
```

文件内容为我们实现的类的全限定名：

```
com.your.package.NacosDriverURLProvider
```

# 第五步: 启动项目, 访问 Nacos 控制台, 配置数据

1. 首先确保 Nacos 服务已启动
2. 登录 Nacos 控制台（通常是 http://localhost:8848/nacos/）
3. 创建配置：
   - Data ID: `live-user-shardingjdbc.yaml`
   - Group: `DEFAULT_GROUP`
   - 格式: YAML
   - 内容: 粘贴上文中的 ShardingJDBC 配置

4. 同样方式创建 `live-user-provider.yaml` 配置

## URL 配置说明

在项目的数据源配置中，可以根据需要进行不同的设置：

### 基本配置（不需要认证）

```yaml
url: jdbc:shardingsphere:nacos:127.0.0.1:8848:live-user-shardingjdbc.yaml
```

### 带认证信息的配置

```yaml
url: jdbc:shardingsphere:nacos:127.0.0.1:8848:live-user-shardingjdbc.yaml?username=nacos&&password=nacos&&namespace=live-test
```

## 简化本地配置

只需保留本地的 bootstrap.yaml 配置即可：

```yaml
spring:
  application:
    name: live-user-provider
  cloud:
    nacos:
      #      username: nacos  # 如果 Nacos 需要认证，取消注释
      #      password: nacos  # 如果 Nacos 需要认证，取消注释
      discovery:
        server-addr: localhost:8848
      config:
        import-check:
          enabled: false
        file-extension: yaml
        # 读取配置的 nacos 地址
        server-addr: localhost:8848
  config:
    import:
      - optional:nacos:live-user-provider.yaml
```

# 常见问题及解决方案

## 1. Nacos 连接超时

**问题**: 启动项目时报 Nacos 连接超时错误。

**解决方案**: 
- 检查 Nacos 服务是否正常运行
- 检查网络连接是否通畅
- 在代码中适当增加连接超时时间
- 确认防火墙、安全组等设置不会阻止连接

## 2. 配置内容解析错误

**问题**: ShardingJDBC 报告配置格式不正确的错误。

**解决方案**:
- 验证 YAML 格式是否正确，使用在线 YAML 校验工具检查
- 确保配置符合 ShardingJDBC 5.3.2 版本的规范
- 检查配置中的路径、表名等引用是否正确

## 3. 找不到 SPI 实现类

**问题**: ShardingJDBC 无法找到我们的 `NacosDriverURLProvider` 实现。

**解决方案**:
- 确保 `META-INF/services` 目录结构正确
- 确保服务文件名为接口的完整路径名
- 检查实现类的全限定名是否正确
- 检查项目打包后是否包含了 SPI 配置文件

# 总结与拓展

通过本文的方法，我们成功地将 ShardingJDBC 的配置迁移到了 Nacos，这样做有以下优势：

1. **集中配置管理**: 所有服务的配置可以统一在 Nacos 管理
2. **动态配置更新**: 可以实现配置的动态更新（需要配合 ShardingJDBC 的配置刷新机制）
3. **环境隔离**: 可以通过 Nacos 的 Namespace 实现不同环境的配置隔离
4. **版本管理**: Nacos 支持配置的历史版本和回滚功能
5. **配置加密**: 敏感配置可以通过 Nacos 的配置加密功能保护

## 拓展思路

1. **配置动态更新**: 可以结合 ShardingJDBC 的配置刷新机制，实现配置变更后的自动更新
2. **多环境支持**: 利用 Nacos 的 Namespace 和 Group 机制，实现开发、测试、生产环境的配置隔离
3. **配置审计**: 实现配置变更的审计日志，记录谁在什么时间修改了配置
4. **配置中心扩展**: 除了 Nacos，还可以支持 Apollo、Consul 等其他配置中心

通过以上实践，可以更加灵活地管理分布式系统中的 ShardingJDBC 配置，提高系统的可维护性和可扩展性。