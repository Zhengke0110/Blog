---
title: 关于SearXNG无法返回JSON的问题
date: 2025-09-23
type: record
---

# Docker 部署 SearXNG

```bash
docker pull searxng/searxng:latest

docker run -p 18080:8080 \
        --name searxng \
        -d --restart=always \
        -v "电脑上挂载的地址/docker/SearXNG:/etc/searxng" \
        -e "BASE_URL=http://localhost:$PORT/" \
        -e "INSTANCE_NAME=instance" \
        searxng/searxng
```

执行搜索

```bash
curl -v "http://localhost:18080/search?q=今夕是何年&format=json"
```

报错如下

```bash


* Host localhost:18080 was resolved.
* IPv6: ::1
* IPv4: 127.0.0.1
*   Trying [::1]:18080...
* Connected to localhost (::1) port 18080
> GET /search?q=风间影月&format=json HTTP/1.1
> Host: localhost:18080
> User-Agent: curl/8.7.1
> Accept: */*
>
* Request completely sent off
< HTTP/1.1 403 Forbidden
< content-type: text/html; charset=utf-8
< content-length: 213
< server-timing: total;dur=2.396, render;dur=0
< x-content-type-options: nosniff
< x-download-options: noopen
< x-robots-tag: noindex, nofollow
< referrer-policy: no-referrer
< server: granian
< date: Tue, 23 Sep 2025 06:09:49 GMT
<
<!doctype html>
<html lang=en>
<title>403 Forbidden</title>
<h1>Forbidden</h1>
<p>You don&#39;t have the permission to access the requested resource. It is either read-protected or not readable by the server.</p>
* Connection #0 to host localhost left intact

```

# 解决方案

在配置文件中，formats 部分只启用了 html 格式，需要额外启用 json

1. 找到 `searxng/settings.yml` 文件
2. 修改 `formats:` 为 `formats: [html, json]`
3. 重启容器
