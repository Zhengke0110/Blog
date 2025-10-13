---
title: Docker 容器访问宿主机 Ollama 服务配置
date: 2025-09-28
type: notes
---

## 问题背景

在使用 Docker 部署 Dify 时遇到 Ollama 连接错误：

```
HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries exceeded with url: /api/chat (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0xffff985bd670>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

## 问题原因

Docker 容器内部无法通过 `127.0.0.1:11434` 访问宿主机上运行的 Ollama 服务。需要使用 `host.docker.internal:11434` 来访问宿主机服务。

## 解决步骤

### 1. 检查 Ollama 服务状态

```bash
# 检查Ollama进程是否运行
ps aux | grep ollama

# 检查端口11434是否被占用
lsof -i :11434

# 测试Ollama API连接
curl -s http://127.0.0.1:11434/api/version
```

### 2. 创建环境配置文件

```bash
# 进入Dify项目目录
cd /Users/zhengke/Downloads/dify/dify-docker

# 复制示例配置文件
cp .env.example .env
```

### 3. 修改 Ollama 配置

编辑 `.env` 文件，修改 `OPENAI_API_BASE` 配置：

```bash
# 原配置
OPENAI_API_BASE=https://api.openai.com/v1

# 修改为
OPENAI_API_BASE=http://host.docker.internal:11434/v1
```

### 4. 重启 Docker 容器

```bash
# 停止所有容器
docker-compose down

# 启动所有容器
docker-compose up -d
```

### 5. 验证配置

```bash
# 测试容器内是否能访问Ollama
docker exec -it dify-docker-api-1 curl -s http://host.docker.internal:11434/api/version

# 检查环境变量是否正确设置
docker exec dify-docker-api-1 env | grep OPENAI_API_BASE

# 查看API容器日志
docker logs dify-docker-api-1 --tail 20
```

## 关键配置说明

### Docker 网络访问

- **容器内访问宿主机**: 使用 `host.docker.internal` 替代 `127.0.0.1` 或 `localhost`
- **端口映射**: Ollama 默认监听 `11434` 端口
- **API 兼容性**: Ollama 提供 OpenAI 兼容的 API 端点 `/v1`

### 环境变量配置

```bash
# 关键环境变量
OPENAI_API_BASE=http://host.docker.internal:11434/v1

# 其他重要配置
VECTOR_STORE=weaviate  # 向量数据库选择
DB_USERNAME=postgres   # 数据库用户名
DB_PASSWORD=difyai123456  # 数据库密码
REDIS_PASSWORD=difyai123456  # Redis密码
```

### Docker Compose 服务

主要服务包括：

- `api`: Dify API 服务
- `worker`: 队列处理服务
- `web`: 前端 Web 界面
- `db`: PostgreSQL 数据库
- `redis`: Redis 缓存
- `weaviate`: 向量数据库
- `sandbox`: 代码执行沙盒
- `nginx`: 反向代理

## 故障排查指南

### 1. 检查服务状态

```bash
# 查看所有容器状态
docker-compose ps

# 查看特定容器日志
docker logs dify-docker-api-1
docker logs dify-docker-worker-1
docker logs dify-docker-nginx-1
```

### 2. 网络连通性测试

```bash
# 从容器内测试连接
docker exec -it dify-docker-api-1 curl -v http://host.docker.internal:11434/api/version

# 测试Dify Web界面
curl -s -o /dev/null -w "%{http_code}" http://localhost
```

### 3. 常见问题

1. **端口冲突**: 确保 80、443、5001 等端口未被占用
2. **内存不足**: 确保系统有足够内存运行所有服务
3. **权限问题**: 确保 Docker 有足够权限访问挂载目录

## 访问地址

- **Dify Web 界面**: http://localhost
- **API 文档**: http://localhost/swagger-ui.html
- **Ollama API**: http://localhost:11434

## 注意事项

1. 首次启动可能需要几分钟时间初始化数据库
2. 确保 Ollama 服务在 Dify 启动前已经运行
3. 生产环境建议修改默认密码和密钥
4. 定期备份数据库和存储卷

## 相关文件

- `docker-compose.yaml`: Docker 服务编排配置
- `.env`: 环境变量配置
- `volumes/`: 数据持久化目录

---

_创建时间: 2025 年 9 月 28 日_
_问题解决: Docker 容器访问宿主机 Ollama 服务配置_
