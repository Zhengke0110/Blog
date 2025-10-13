---
title: 前端页面 Set Cookie 失败
date: 2025-01-5
type: notes
---

在 SpringBoot 项目中添加配置文件

```java
@Configuration
@EnableSpringHttpSession
public class SessionConfig {

    @Bean
    public SessionRepository sessionRepository() {
        return new MapSessionRepository(new ConcurrentHashMap<>());
    }

    @Bean
    DefaultCookieSerializerCustomizer cookieSerializerCustomizer() {
        return new DefaultCookieSerializerCustomizer() {
            @Override
            public void customize(DefaultCookieSerializer cookieSerializer) {
                cookieSerializer.setSameSite("None"); // 设置cookie的SameSite属性为None，否则跨域set-cookie会被chrome浏览器阻拦
                cookieSerializer.setUseSecureCookie(true); // sameSite为None时，useSecureCookie必须为true
            }
        };
    }
}
```
