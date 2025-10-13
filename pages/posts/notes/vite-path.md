---
title: Vite vue-ts 配置'@'路径别名
date: 2022-04-14
type: notes
---

1. 安装 node 依赖(类型)

```bash
yarn add @types/node -D
```

2. 修改 vite.config.js

```js
import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig {
    // ...
    resolve: {
        alias: {
            "@": resolve(__dirname, 'src'), // 路径别名
        },
        extensions: ['.js', '.json', '.ts'] // 使用路径别名时想要省略的后缀名,可以自己 增减
    }
    // ...
}
```

3. 修改 tsconfig.json

```json
{
  "compilerOptions": {
    // ...
    "baseUrl": ".", // 用于设置解析非相对模块名称的基本目录，相对模块不会受到baseUrl的影响
    "paths": {
      // 用于设置模块名到基于baseUrl的路径映射
      "@/*": ["src/*"]
    }
    // ...
  }
}
```
