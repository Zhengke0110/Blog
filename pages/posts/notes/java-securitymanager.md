---
title: Java17 安全管理器 被弃用的解决方案(JAVM参数调用的方式)
date: 2025-01-14
type: notes
---

### 简要解释报错原因

System.setSecurityManager 方法自 Java 17 起被标记为已弃用，并计划在未来版本中移除。这是因为 Java 安全管理器（Security Manager）本身在未来的 Java 版本中将不再支持，官方推荐使用其他安全机制来替代。

### 使用 JVM 参数限制资源

在 Java 中，可以通过设置 JVM 参数来配置安全策略和资源限制。具体来说，`java.security.manager` 和 `java.security.policy` 是用于控制 Java 应用程序的安全性和权限的两个重要参数。

#### 1. 设置 `java.security.manager`

- **作用**：启用 Java 安全管理器（Security Manager）。虽然 `System.setSecurityManager` 方法已被弃用，但通过 JVM 参数仍然可以启用安全管理器。
- **使用方法**：在启动应用程序时，通过命令行参数 `-Djava.security.manager=allow` 来启用安全管理器。`allow` 表示允许使用安全管理器，而不会触发默认的安全策略。

#### 2. 设置 `java.security.policy`

- **作用**：指定一个安全策略文件，该文件定义了哪些代码可以执行哪些操作。通过这个文件，可以细粒度地控制代码的权限。
- **使用方法**：在启动应用程序时，通过命令行参数 `-Djava.security.policy=path/to/your/policy/file` 指定安全策略文件的路径。

#### 3. 创建安全策略文件

安全策略文件是一个文本文件，通常以 `.policy` 为扩展名。它定义了哪些代码具有哪些权限。以下是一个简单的安全策略文件示例：

```plaintext
grant {
    // 允许所有代码执行基本的文件读写操作
    permission java.io.FilePermission "<<ALL FILES>>", "read,write";

    // 允许所有代码执行网络连接
    permission java.net.SocketPermission "*", "connect,resolve";

    // 允许所有代码执行系统退出操作
    permission java.lang.RuntimePermission "exitVM";
};
```

你可以根据实际需求调整这些权限。例如，如果你希望限制代码只能读取特定目录下的文件，可以将 `<<ALL FILES>>` 替换为具体的路径。

#### 4. 修改启动脚本或 IDE 配置

为了使这些参数生效，你需要在启动应用程序时传递这些参数。以下是几种常见的方式：

- **命令行启动**：

  ```bash
  java -Djava.security.manager=allow -Djava.security.policy=/path/to/your/policy/file -jar your-application.jar
  ```

- **IDE 配置**（如 IntelliJ IDEA 或 Eclipse）：
  在运行配置中添加 VM 参数：

  ```
  -Djava.security.manager=allow -Djava.security.policy=/path/to/your/policy/file
  ```

- **Maven 或 Gradle 构建工具**：
  如果你使用 Maven 或 Gradle 进行构建，可以在相应的配置文件中添加这些参数。例如，在 Maven 的 `pom.xml` 中：

  ```xml
  <build>
      <plugins>
          <plugin>
              <groupId>org.apache.maven.plugins</groupId>
              <artifactId>maven-surefire-plugin</artifactId>
              <version>3.0.0-M5</version>
              <configuration>
                  <argLine>-Djava.security.manager=allow -Djava.security.policy=/path/to/your/policy/file</argLine>
              </configuration>
          </plugin>
      </plugins>
  </build>
  ```

#### 5. 示例代码中的应用

在你的 `JavaNativeCodeSandbox.java` 文件中，不需要直接设置这些属性，而是通过上述方式在启动应用程序时传递这些参数。因此，你可以移除或注释掉相关代码：

```java
// 移除或注释掉这两行
// System.setProperty("java.security.manager", "allow");
// System.setProperty("java.security.policy", "path/to/your/policy/file");
```

### 总结

通过设置 JVM 参数 `java.security.manager` 和 `java.security.policy`，你可以在不使用已弃用的 `System.setSecurityManager` 方法的情况下，实现对代码执行环境的安全控制。确保创建并配置合适的安全策略文件，并在启动应用程序时正确传递这些参数。

### 示例

```java
public class MySecurityManager extends SecurityManager {

    // 日志记录器，用于记录安全相关的操作信息
    private static final Logger logger = LoggerFactory.getLogger(MySecurityManager.class);

    // 允许访问的文件路径前缀，用于限制文件访问范围
    private static final String ALLOWED_PATH = System.getProperty("allowed.path", "Path");

    /**
     * 检查所有的权限
     * 此方法重写自父类，用于在运行时检查给定的权限是否可以被授予
     *
     * @param perm 要检查的权限对象
     */
    @Override
    public void checkPermission(Permission perm) {
        super.checkPermission(perm);
    }

    /**
     * 检测程序是否可执行文件
     * 重写此方法以禁止执行任何命令
     *
     * @param cmd 待执行的命令
     * @throws SecurityException 如果命令为空或执行命令被拒绝
     */
    @Override
    public void checkExec(String cmd) {
        if (cmd == null || cmd.trim().isEmpty()) {
            throw new SecurityException("命令不能为空");
        }
        throw new SecurityException("checkExec 权限异常：命令执行被拒绝");
    }

    /**
     * 检测程序是否允许读文件
     * 重写此方法以限制文件读取操作，仅允许读取特定路径下的文件
     *
     * @param file 文件路径
     * @throws SecurityException 如果文件路径为空或尝试读取的文件不在允许的路径下
     */
    @Override
    public void checkRead(String file) {
        if (file == null || file.trim().isEmpty()) {
            throw new SecurityException("文件路径不能为空");
        }
        File filePath = new File(file);
        if (!filePath.getAbsolutePath().startsWith(ALLOWED_PATH)) {
            throw new SecurityException("checkRead 权限异常：不允许读取此文件");
        }
        logger.info("允许读取文件: {}", file);
    }

    /**
     * 检测程序是否允许写文件
     * 重写此方法以禁止写入任何文件
     *
     * @param file 文件路径
     * @throws SecurityException 如果文件路径为空或尝试写入文件被拒绝
     */
    @Override
    public void checkWrite(String file) {
        if (file == null || file.trim().isEmpty()) {
            throw new SecurityException("文件路径不能为空");
        }
        throw new SecurityException("checkWrite 权限异常：不允许写入此文件");
    }

    /**
     * 检测程序是否允许删除文件
     * 重写此方法以禁止删除任何文件
     *
     * @param file 文件路径
     * @throws SecurityException 如果文件路径为空或尝试删除文件被拒绝
     */
    @Override
    public void checkDelete(String file) {
        if (file == null || file.trim().isEmpty()) {
            throw new SecurityException("文件路径不能为空");
        }
        throw new SecurityException("checkDelete 权限异常：不允许删除此文件");
    }

    /**
     * 检测程序是否允许连接网络
     * 重写此方法以禁止网络连接操作
     *
     * @param host 目标主机名
     * @param port 目标端口号
     * @throws SecurityException 如果主机名为空、端口号无效或网络连接被拒绝
     */
    @Override
    public void checkConnect(String host, int port) {
        if (host == null || host.trim().isEmpty() || port <= 0 || port > 65535) {
            throw new SecurityException("无效的主机或端口");
        }
        throw new SecurityException("checkConnect 权限异常：不允许连接到 " + host + ":" + port);
    }
}
```

### 将 `MySecurityManager` 的权限配置抽取为 `.policy` 文件

为了将 `MySecurityManager` 中定义的权限规则抽取到一个 `.policy` 文件中，你需要明确哪些权限需要授予给特定的代码，并在 `.policy` 文件中进行相应的配置。以下是具体的步骤和示例。

#### 1. 理解 `MySecurityManager` 中的权限控制

从提供的 `MySecurityManager` 类来看，主要的权限控制包括：

- **禁止执行命令** (`checkExec`)
- **限制文件读取路径** (`checkRead`)
- **禁止写入文件** (`checkWrite`)
- **禁止删除文件** (`checkDelete`)
- **禁止网络连接** (`checkConnect`)

这些权限可以通过安全策略文件（`.policy`）来实现更细粒度的控制。

#### 2. 创建安全策略文件

创建一个名为 `security.policy` 的文件，并根据你的需求配置权限。以下是一个示例 `.policy` 文件，假设你希望允许某些包或类执行特定操作，而其他操作仍然受限。

```plaintext
// security.policy 文件内容

// 允许所有代码使用 MySecurityManager
grant {
    permission java.security.AllPermission;
};

// 允许特定包读取文件，仅限于指定路径下的文件
grant codeBase "file:/path/to/your/codebase/-" {
    permission java.io.FilePermission "${allowed.path}${/}-", "read";
};

// 禁止执行任何命令
grant {
    permission java.lang.RuntimePermission "exitVM";
    // 禁止执行命令
    denial java.lang.RuntimePermission "exec";
};

// 禁止写入文件
grant {
    // 禁止写入任何文件
    denial java.io.FilePermission "<<ALL FILES>>", "write";
};

// 禁止删除文件
grant {
    // 禁止删除任何文件
    denial java.io.FilePermission "<<ALL FILES>>", "delete";
};

// 禁止网络连接
grant {
    // 禁止网络连接
    denial java.net.SocketPermission "*", "connect,resolve";
};
```

#### 3. 解释 `.policy` 文件中的关键部分

- **`grant` 块**：定义了哪些代码具有哪些权限。

  - `grant { ... };`：默认情况下，所有代码都具有这些权限。
  - `grant codeBase "file:/path/to/your/codebase/-" { ... };`：仅允许特定代码库中的代码具有某些权限。

- **权限声明**：

  - `java.io.FilePermission "${allowed.path}${/}-", "read";`：允许读取 `${allowed.path}` 目录及其子目录下的文件。
  - `denial java.lang.RuntimePermission "exec";`：显式拒绝执行命令的权限。
  - `denial java.io.FilePermission "<<ALL FILES>>", "write";`：拒绝写入任何文件的权限。
  - `denial java.io.FilePermission "<<ALL FILES>>", "delete";`：拒绝删除任何文件的权限。
  - `denial java.net.SocketPermission "*", "connect,resolve";`：拒绝所有网络连接的权限。

#### 4. 修改启动命令以启用安全管理器

确保在启动应用程序时传递 JVM 参数以启用安全管理器并指定安全策略文件的路径。

**命令行启动：**

```bash
java -Djava.security.manager=allow -Djava.security.policy=/path/to/security.policy -jar your-application.jar
```

#### 5. 配置 IDE 或构建工具

如果你使用的是 IDE（如 IntelliJ IDEA 或 Eclipse）或构建工具（如 Maven 或 Gradle），可以在相应的配置中添加这些参数。

- **IntelliJ IDEA**：

  - 打开运行配置。
  - 在 VM options 中添加：
    ```
    -Djava.security.manager=allow -Djava.security.policy=/path/to/security.policy
    ```

- **Maven**：

  - 在 `pom.xml` 中配置插件参数：

    ```xml
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0-M5</version>
                <configuration>
                    <argLine>-Djava.security.manager=allow -Djava.security.policy=/path/to/security.policy</argLine>
                </configuration>
            </plugin>
        </plugins>
    </build>
    ```

#### 6. 测试和验证

确保在应用这些更改后，测试应用程序的行为是否符合预期。特别是检查文件读取、写入、删除和网络连接等操作是否被正确限制。
