---
title: MySQL主从数据库同步(Docker部署)
date: 2025-03-01
draft: true
lang: zh
---

# 部署主数据库

主数据库配置文件

```bash
[mysqld]
datadir = /xxx/master1/data
# 服务器唯一id，默认值1
server-id=1
# 设置日志格式，默认值ROW
binlog_format=STATEMENT
# 二进制日志名，默认binlog
log-bin=binlog
```

docker 部署主数据库命令

```bash
docker run --name=mysql-master-1 \
--privileged=true \
-p 8808:3306 \
-v /xxx/master1/data/:/var/lib/mysql \
-v /xxx/master1/conf/my.cnf:/etc/mysql/my.cnf \
-v /xxx/master1/mysql-files/:/var/lib/mysql-files/ \
-e MYSQL_ROOT_PASSWORD=root \
-d mysql:8.0 --lower_case_table_names=1
```

# 部署从数据库

从数据库配置文件

```bash
[mysqld]
datadir = /xxx/slave1/data
server-id=2
```

docker 部署从数据库命令

```bash
docker run --name=mysql-slave-1 \
--privileged=true \
-p 8809:3306 \
-v /xxx/slave1/data/:/var/lib/mysql \
-v /xxx/slave1/conf/my.cnf:/etc/mysql/my.cnf \
-v /xxx/slave1/mysql-files/:/var/lib/mysql-files/ \
-e MYSQL_ROOT_PASSWORD=root \
-d mysql:8.0 --lower_case_table_names=1
```

# 配置主从同步

主数据库

```bash
docker exec -it mysql-master-1 /bin/bash

mysql -uroot -proot # 登录

CREATE USER 'slave' @'%' IDENTIFIED WITH mysql_native_password BY '123456'; # 创建用户

GRANT replication SLAVE ON*.*TO 'slave' @'%';  # 授权

flush privileges; # 刷新权限

show variables like 'server_id'; # 查看server_id

show master status; # 查看主库的binlog信息
```
![主数据库]('/public/images/blog/mysql-sync/master.png')


从数据库

```bash
docker exec -it mysql-slave-1 /bin/bash

mysql -uroot -proot # 登录

show variables like 'server_id'; # 查看server_id

set global server_id = 2; # 设置server_id 从数据库需要与主数据库不同

# 若之前设置过同步，请先重置
stop slave;
reset slave;

change master to master_host='这里不能填127.0.0.1',master_port=8808,master_user='slave',master_password='123456',master_log_file='binlog.000001',master_log_pos=801; # 设置主数据库
# 注意:
# master_log_file: 主数据库的binlog文件名(主库执行 show master status; 获取 )
# master_log_pos: 主数据库的binlog文件位置(主库执行 show master status; 获取 )
# master_host: 主数据库的ip地址 需要查看主机ip，不能使用本地环回地址
# master_port: 主数据库的端口号

start slave;

show slave status; # 查询 Slave 状态
```

![成功]('/public/images/blog/mysql-sync/success.png')

**注意: 从库设置时，一定不能使用本地环回地址，这样会出现主从同步失败(Connecting 状态 是个坑)**