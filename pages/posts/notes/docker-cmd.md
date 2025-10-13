---
title: 记录:Docker 创建容器命令
date: 2025-05-13
type: notes
---
[截止2025年5月13日可用Docker镜像源解决方案](https://blog.csdn.net/m0_37899908/article/details/145090710)
## Redis

```bash
# 拉取镜像
docker pull redis

# 运行
docker run -it -d --name redis -p 6379:6379 redis --bind 0.0.0.0 --protected-mode no
```

## MySQL(Windows 路径可进行替换)

```bash
docker pull mysql:8.3.0

docker run --name mysql-8.3.0 -p 13306:3306 -v D:/Service/mysql/docker/data:/var/lib/mysql -v D:/Service/mysql/docker/logs:/var/log/mysql -v D:/Service/mysql/docker/conf:/etc/mysql/conf.d -e MYSQL_ROOT_PASSWORD=123456 --restart=always --privileged=true -d mysql:8.3.0
```

## RabbitMQ

```bash

docker pull rabbitmq:management

docker run -dit --name rabbitmq -e RABBITMQ_DEFAULT_USER=admin -e RABBITMQ_DEFAULT_PASS=admin -v D:/Service/rabbitmq/data:/var/lib/rabbitmq/mnesia/ -p 15672:15672 -p 5672:5672 rabbitmq:management
```

## RocketMQ

```bash
docker pull apache/rocketmq:5.3.2

docker network create rocketmq

# 启动 NameServer
docker run -d --name rmqnamesrv -p 9876:9876 --network rocketmq apache/rocketmq:5.3.2 sh mqnamesrv

# 验证 NameServer 是否启动成功
docker logs -f rmqnamesrv

# 配置 Broker 的 IP 地址
echo "brokerIP1=127.0.0.1" > broker.conf

# 启动 Broker 和 Proxy
docker run -d --name rmqbroker  --net rocketmq  -p 10911:10911 -p 10909:10909 -p 10912:10912  -p 8080:8080 -p 8081:8081  -e "NAMESRV_ADDR=rmqnamesrv:9876"  -v "C:/Users/PC/broker.conf:/home/rocketmq/rocketmq-5.3.2/conf/broker.conf"  apache/rocketmq:5.3.2  sh mqbroker -c /home/rocketmq/rocketmq-5.3.2/conf/broker.conf

# 验证 Broker 是否启动成功
docker exec -it rmqbroker bash -c "tail -n 10 /home/rocketmq/logs/rocketmqlogs/proxy.log"
```

## Clickhouse

```bash
docker pull yandex/clickhouse-server
docker pull yandex/clickhouse-client

docker run --rm -d --name=temp-clickhouse-server yandex/clickhouse-server

docker cp temp-clickhouse-server:/etc/clickhouse-server/config.xml D:/Service/clickhouse/conf/config.xml

docker cp temp-clickhouse-server:/etc/clickhouse-server/users.xml D:/Service/clickhouse/conf/users.xml


docker run -d --name=single-clickhouse-server -p 8123:8123 -p 9000:9000 -p 9009:9009 --ulimit nofile=262144:262144 --volume D:/Service/clickhouse/data:/var/lib/clickhouse:rw --volume D:/Service/clickhouse/conf:/etc/clickhouse-server:rw --volume D:/Service/clickhouse/log:/var/log/clickhouse-server:rw yandex/clickhouse-server
```

## kafka（非 zookeeper）

```bash

docker pull apache/kafka:3.9.0

docker run --privileged=true --net=bridge -d --name=kafka-kraft -v D:/Service/kafka-kraft/data:/var/lib/kafka/data -v D:/Service/kafka-kraft/config:/mnt/shared/config -v D:/Service/kafka-kraft/secrets:/etc/kafka/secrets -p 9092:9092 -p 9093:9093 -e TZ=Asia/Shanghai -e LANG=C.UTF-8 -e KAFKA_NODE_ID=1 -e CLUSTER_ID=kafka-cluster -e KAFKA_PROCESS_ROLES=broker,controller -e KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://192.168.0.103:9092 -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 apache/kafka:3.9.0
```
## Kafka

```bash

docker pull wurstmeister/zookeeper
docker pull wurstmeister/kafka

docker run -d --name zookeeper -p 2181 -t wurstmeister/zookeeper

docker run -d --name kafka --publish 9092:9092 --link zookeeper --env KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 --env KAFKA_ADVERTISED_HOST_NAME=127.0.0.1 --env KAFKA_ADVERTISED_PORT=9092 wurstmeister/kafka:latest

```