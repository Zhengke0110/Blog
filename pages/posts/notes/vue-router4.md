---
title: Vue Router 4 指定未识别的路径报错
date: 2022-08-18
type: notes
---

老版本写法:

```ts
 {
    path: "*",
    redirect: "/main",
  },
```

新版本写法:

```ts
 {
    path: "/:catchAll(.*)",
    redirect: "/main",
  },
```
