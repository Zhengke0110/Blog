---
title: MySQL主从数据库同步(Docker部署)
date: 2025-03-01
draft: true
lang: zh
---

# MySQL主从同步原理与Docker实现

MySQL主从复制（Replication）是一种常见的数据库高可用性解决方案，它允许数据从一个MySQL数据库服务器（主服务器）复制到一个或多个MySQL数据库服务器（从服务器）。主从复制不仅提高了数据的安全性，还可以实现负载均衡，提升整体系统性能。

本文将介绍如何使用Docker容器快速搭建MySQL主从同步环境，实现数据的实时复制。

## 主从复制原理

MySQL主从复制的基本原理：

1. **主库记录binlog**：当主库上发生数据修改操作时，会记录到二进制日志（binary log）中
2. **从库读取binlog**：从库通过I/O线程请求主库的binlog，并将其拷贝到本地的中继日志（relay log）
3. **从库回放日志**：从库通过SQL线程读取中继日志中的事件，并在本地重放，从而实现数据同步

![MySQL复制原理](/images/blog/mysql-sync/replication-diagram.png)

# 部署主数据库

## 主数据库配置文件详解

创建主数据库配置文件（my.cnf）：

```bash
[mysqld]
datadir = /xxx/master1/data
# 服务器唯一id，默认值1，在复制拓扑中必须唯一
server-id=1
# 设置日志格式，可选值：STATEMENT, ROW, MIXED
# STATEMENT: 记录SQL语句，可能导致主从不一致
# ROW: 记录行数据变化，更安全但日志量大
# MIXED: MySQL自动选择合适的格式
binlog_format=STATEMENT
# 二进制日志文件名前缀
log-bin=binlog
# 启用GTID模式（可选，MySQL 5.6及以上版本支持）
# gtid_mode=ON
# enforce_gtid_consistency=ON
# 指定需要复制的数据库（可选）
# binlog-do-db=db_name
# 指定不需要复制的数据库（可选）
# binlog-ignore-db=mysql
```

## docker 部署主数据库命令详解

```bash
docker run --name=mysql-master-1 \
--privileged=true \                    # 给予容器特权模式
-p 8808:3306 \                         # 映射端口，主机8808映射到容器3306
-v /xxx/master1/data/:/var/lib/mysql \ # 挂载数据目录
-v /xxx/master1/conf/my.cnf:/etc/mysql/my.cnf \ # 挂载配置文件
-v /xxx/master1/mysql-files/:/var/lib/mysql-files/ \ # 挂载文件目录
-e MYSQL_ROOT_PASSWORD=root \          # 设置root密码
-d mysql:8.0 \                         # 使用mysql:8.0镜像后台运行
--lower_case_table_names=1             # 设置表名不区分大小写
```

**注意**：在实际部署中，请将 `/xxx/` 替换为您自己的实际目录路径。

# 部署从数据库

## 从数据库配置文件详解

```bash
[mysqld]
datadir = /xxx/slave1/data
# 服务器唯一id，必须与主库不同
server-id=2
# 启用中继日志，记录从主库复制的事件
relay-log=slave-relay-bin
# 启用从库更新日志（可选）
log-bin=slave-bin
# 允许从库将更改写入二进制日志（用于级联复制）
log-slave-updates=1
# 只读模式，防止意外写入
read_only=1
```

## docker 部署从数据库命令详解

```bash
docker run --name=mysql-slave-1 \
--privileged=true \                   # 给予容器特权模式
-p 8809:3306 \                        # 映射端口，主机8809映射到容器3306
-v /xxx/slave1/data/:/var/lib/mysql \ # 挂载数据目录  
-v /xxx/slave1/conf/my.cnf:/etc/mysql/my.cnf \ # 挂载配置文件
-v /xxx/slave1/mysql-files/:/var/lib/mysql-files/ \ # 挂载文件目录
-e MYSQL_ROOT_PASSWORD=root \         # 设置root密码
-d mysql:8.0 \                        # 使用mysql:8.0镜像后台运行
--lower_case_table_names=1            # 设置表名不区分大小写
```

# 配置主从同步

## 主数据库配置

1. 首先，进入主数据库容器：

```bash
docker exec -it mysql-master-1 /bin/bash
```

2. 登录MySQL：

```bash
mysql -uroot -proot # 登录
```

3. 创建用于复制的专用账户：

```bash
# 创建用于复制的用户，推荐创建专用用户而非使用root
CREATE USER 'slave'@'%' IDENTIFIED WITH mysql_native_password BY '123456';

# 授予复制所需的权限
GRANT REPLICATION SLAVE ON *.* TO 'slave'@'%';

# 刷新权限
FLUSH PRIVILEGES;

# 确认服务器ID设置正确
SHOW VARIABLES LIKE 'server_id';

# 查看主库状态，记下File和Position的值，后续配置从库时需要
SHOW MASTER STATUS;
```

输出结果类似于：

![主数据库状态](/images/blog/mysql-sync/master.png)

**重要提示**：记录下 `File` 和 `Position` 的值，配置从库时需要用到。

## 从数据库配置

1. 进入从数据库容器：

```bash
docker exec -it mysql-slave-1 /bin/bash
```

2. 登录MySQL：

```bash
mysql -uroot -proot # 登录
```

3. 配置复制：

```bash
# 确认服务器ID设置正确且与主库不同
SHOW VARIABLES LIKE 'server_id';

# 如果server_id未正确设置，可以动态修改
SET GLOBAL server_id = 2;

# 如果之前配置过复制，需要先停止并重置
STOP SLAVE;
RESET SLAVE;

# 配置主库连接信息
CHANGE MASTER TO 
    MASTER_HOST='172.17.0.2',        # 主库容器IP，不能使用127.0.0.1
    MASTER_PORT=3306,                # 主库MySQL端口（容器内端口）
    MASTER_USER='slave',             # 主库复制用户名
    MASTER_PASSWORD='123456',        # 主库复制用户密码
    MASTER_LOG_FILE='binlog.000001', # 主库二进制日志文件名（从SHOW MASTER STATUS获取）
    MASTER_LOG_POS=801;              # 主库二进制日志位置（从SHOW MASTER STATUS获取）

# 启动从库复制线程
START SLAVE;

# 查看从库状态
SHOW SLAVE STATUS\G
```

**注意事项**：
1. `MASTER_HOST` 必须使用主数据库容器的实际IP地址，而不能使用`127.0.0.1`或`localhost`。可以通过`docker inspect mysql-master-1 | grep IPAddress`查看主库容器IP。
2. `MASTER_PORT` 应该是容器内的MySQL端口（默认3306），不是宿主机映射端口。
3. `MASTER_LOG_FILE` 和 `MASTER_LOG_POS` 必须与主库的 `SHOW MASTER STATUS` 输出一致。

正确配置后，`SHOW SLAVE STATUS` 输出中应该能看到：

![从库状态成功](/images/blog/mysql-sync/success.png)

确认以下两项状态为Yes表示复制正常运行：
- `Slave_IO_Running: Yes`
- `Slave_SQL_Running: Yes`

# 测试主从同步

验证主从同步是否正常工作：

1. 在主库创建数据库和表：

```sql
-- 在主库执行
CREATE DATABASE test_db;
USE test_db;
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(50),
  create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO users (name) VALUES ('user1'), ('user2');
```

2. 在从库检查数据是否同步：

```sql
-- 在从库执行
SHOW DATABASES;
USE test_db;
SHOW TABLES;
SELECT * FROM users;
```

如果能在从库看到主库创建的数据库、表和数据，说明主从同步配置成功。

# 常见问题与故障排除

## 1. 从库复制状态显示"Connecting"

**问题**：`Slave_IO_Running` 显示 "Connecting" 而不是 "Yes"。

**可能原因**：
- 网络连接问题
- 主库地址配置错误（使用了127.0.0.1）
- 认证失败

**解决方案**：
- 确认主从容器网络互通：`ping [主库IP]`
- 检查主库地址配置：使用`docker inspect mysql-master-1 | grep IPAddress`获取正确IP
- 检查复制账户权限：确保在主库上正确创建并授权

## 2. 复制错误

**问题**：`Slave_SQL_Running` 显示 "No"。

**查看错误**：
```sql
SHOW SLAVE STATUS\G
```
检查 `Last_SQL_Error` 字段的错误信息。

**常见解决方案**：
```sql
-- 跳过当前错误（谨慎使用，可能导致数据不一致）
STOP SLAVE;
SET GLOBAL SQL_SLAVE_SKIP_COUNTER = 1;
START SLAVE;
```

## 3. 二进制日志格式问题

如果主库和从库的表结构不完全一致，使用STATEMENT格式可能导致复制错误。考虑改用ROW格式：

```sql
-- 在主库执行
SET GLOBAL binlog_format = 'ROW';
-- 修改配置文件并重启以持久化设置
```

# 总结

正确配置MySQL主从复制可以提高系统的可用性和性能。通过Docker容器部署可以快速搭建测试环境，但在生产环境中还需要考虑数据一致性、故障转移等更多因素。

记住以下关键点：
1. 确保主从服务器的server-id不同
2. 使用正确的网络连接参数
3. 定期检查复制状态
4. 根据业务需求选择适当的二进制日志格式
