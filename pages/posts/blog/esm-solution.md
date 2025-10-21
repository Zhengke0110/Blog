---
title: 解决 "This package is ESM only" 问题
date: 2025-10-21
lang: zh
---

## 问题描述

在 Electron + Vite + Vue 项目中安装 `@vitejs/plugin-vue` 时遇到以下错误：

```bash
✘ [ERROR] "@vitejs/plugin-vue" resolved to an ESM file. ESM file cannot be loaded by `require`.
See https://vite.dev/guide/troubleshooting.html#this-package-is-esm-only for more details.
```

```bash
Build failed with 1 error:
ERROR: [plugin: externalize-deps] "@vitejs/plugin-vue" resolved to an ESM file.
ESM file cannot be loaded by `require`. See https://vite.dev/guide/troubleshooting.html#this-package-is-esm-only for more details.
```

## 问题原因

- `@vitejs/plugin-vue` 是一个 **ESM-only** 包
- 在 Node.js ≤22 中，ESM 文件默认无法通过 `require()` 加载
- Vite 配置文件默认使用 CommonJS 模块系统，无法导入 ESM 模块

## 官方解决方案

根据 [Vite 官方文档](https://vite.dev/guide/troubleshooting.html#this-package-is-esm-only)，有两种解决方案：

### 方案 1: 添加 "type": "module" 到 package.json

```json
{
  "type": "module"
}
```

### 方案 2: 重命名配置文件为 .mjs/.mts 扩展名

- `vite.config.js` → `vite.config.mjs`
- `vite.config.ts` → `vite.config.mts`

## 我们采用的解决方案

对于 **Electron + Vite** 项目，我们选择了 **方案 2**，原因：

1. **避免全局影响**: 不会影响 Electron 主进程的 CommonJS 模块系统
2. **精准解决**: 只针对 Vite 配置文件使用 ESM
3. **兼容性更好**: 保持项目其他部分的模块系统不变

## 实施步骤

### 第一步: 重命名 Vite 配置文件

```bash
# 将 TypeScript 配置文件重命名为 .mts
mv vite.renderer.config.ts vite.renderer.config.mts
```

### 第二步: 更新 Electron Forge 配置

在 `forge.config.ts` 中更新配置文件引用：

```typescript
// 修改前
renderer: [
  {
    name: 'main_window',
    config: 'vite.renderer.config.ts',  // 旧文件名
  },
],

// 修改后
renderer: [
  {
    name: 'main_window',
    config: 'vite.renderer.config.mts', // 新文件名
  },
],
```

### 第三步: 验证配置

`vite.renderer.config.mts` 文件内容：

```typescript
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue"; // 现在可以正常导入 ESM 模块

// https://vitejs.dev/config
export default defineConfig({
  plugins: [vue()],
});
```

## 结果验证

执行 `npm start` 命令，应用成功启动，不再出现 ESM 相关错误。

## 相关文件清单

### 新建/修改的文件

- `vite.renderer.config.mts` (重命名自 `.ts`)
- `forge.config.ts` (更新配置引用)
- `src/App.vue` (Vue 主组件)
- `src/renderer.ts` (初始化 Vue 应用)
- `src/shims-vue.d.ts` (Vue TypeScript 类型声明)
- `index.html` (添加 Vue 挂载点)

### 安装的依赖

```json
{
  "dependencies": {
    "vue": "^3.x.x"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.x.x"
  }
}
```

## TypeScript 配置优化

同时更新了 `tsconfig.json` 以支持现代模块解析：

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext", // 改为 ESNext
    "moduleResolution": "bundler", // 改为 bundler
    "jsx": "preserve" // 支持 Vue JSX
    // ... 其他配置
  }
}
```

## 最佳实践建议

1. **优先选择方案 2**: 对于混合模块系统的项目（如 Electron），重命名配置文件比全局设置更安全
2. **保持一致性**: 如果有多个 Vite 配置文件，都应该重命名为 `.mts`
3. **更新引用**: 记得在所有引用配置文件的地方更新文件名
4. **版本兼容**: 确保使用的 Node.js 版本支持 ESM（建议 Node.js 16+）

## 项目现状

经过上述解决方案，项目现在具备：

- Electron 桌面应用框架
- Vite 现代构建工具
- Vue 3 响应式框架
- TypeScript 类型安全
- ESM 模块兼容性

---

**参考资料:**

- [Vite 官方故障排除文档](https://vite.dev/guide/troubleshooting.html#this-package-is-esm-only)
- [Node.js ESM 文档](https://nodejs.org/docs/latest-v22.x/api/esm.html)
- [Electron Forge 文档](https://www.electronforge.io/)
