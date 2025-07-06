---
title: Vue3 默认图片无限加载问题
date: 2025-02-12
draft: true
lang: zh
---

[解决图片重复加载问题的 Commit](https://github.com/Zhengke0110/MyPictures/commit/6cbd76b5d3324c02b29db2422551934ef5efb238)

## 问题描述

在云图库应用的 Manager Pictures 页面中，某些图片在部署后会无限循环加载。这个问题与默认图片的处理和图片加载错误有关。

![问题描述](/images/blog/bug/2025-2-12-error.png)

## 分析

### 1. 图片加载问题

无限加载问题可能是由 `handleImageError` 函数及其处理图片错误的方式引起的：

```typescript
const handleImageError = (event: Event) => {
  const imgElement = event.target as HTMLImageElement;
  imgElement.src = defaultPlaceholderImage; // 这里可能造成循环
  imgElement.classList.add("opacity-50");
};
```

如果 `defaultPlaceholderImage` 加载失败，会再次触发错误事件，导致无限循环。为了避免这种情况，可以添加一个检查来防止循环：

```typescript
const handleImageError = (event: Event) => {
  const imgElement = event.target as HTMLImageElement;
  if (imgElement.src !== defaultPlaceholderImage) {
    imgElement.src = defaultPlaceholderImage;
    imgElement.classList.add("opacity-50");
  }
};
```

### 2. 缓存机制潜在问题

- `requestCache` 使用 `Map` 存储数据，但没有大小限制和清理机制，可能导致内存泄漏。
- 缓存键使用序列化的 `queryParams` 字符串，可能过于敏感，导致缓存命中率低。

### 3. 性能优化建议

- **懒加载**：对列表中的图片实现懒加载。
- **虚拟滚动**：当处理大量数据时，使用虚拟滚动优化渲染性能。
- **计算属性优化**：优化计算属性，避免频繁重新计算。

### 4. 列表渲染优化

```vue
<tr v-for="item in processedListInfo"
    :key="item.id"
    v-memo="[item.id, item.name, item.thumbnailUrl, item.category, item.tags?.length]"
>
```

`v-memo` 数组包含了许多属性，可能影响性能。建议只保留必要的属性。

### 5. 防抖/节流优化

- `LoadList` 函数使用了 500ms 的节流间隔，可能需要根据实际情况调整。
- `handleKeyPress` 函数使用了 1000ms 的节流时间，可能影响用户体验。

### 6. 监听器优化

```typescript
watch(() => ({
  name: PageInfo.value.name,
  current: PageInfo.value.current,
  // ...
}),
```

这种写法每次都会创建新对象，可能导致不必要的触发。建议分别监听必要的属性。

### 7. 全屏功能潜在问题

```typescript
const toggleFullscreen = (id: string | undefined) => {
  // ...
  if (item.url) {
    imageElement.src = item.url;
    const { toggle } = useFullscreen(imageElement);
    toggle();
    // ...
  }
};
```

在切换全屏时直接修改 `src` 属性可能导致闪烁。建议预加载原图。

### 8. 内存管理

- `imageRefs` 在组件卸载时没有清理。
- 事件监听器在组件卸载时没有移除。

## 解决方案

### 解决方案步骤

1. 修复无限加载问题。
2. 优化图片加载和错误处理逻辑。
3. 添加图片加载状态指示器。
4. 优化缓存机制。

### 代码修改

```typescript
// filepath: /Users/zhengke/Documents/Code/MyPictures/snake-fronted/src/views/pictures/ManagerPictures.vue

// ...existing code...

// 添加新的图片加载状态
const imageLoadingStates = ref<{ [key: string]: boolean }>({});
const failedImages = ref<Set<string>>(new Set());

// 修改图片错误处理函数
const handleImageError = (event: Event, itemId: string) => {
  const imgElement = event.target as HTMLImageElement;
  const currentSrc = imgElement.src;

  if (currentSrc === defaultPlaceholderImage || failedImages.value.has(itemId)) {
    return;
  }

  failedImages.value.add(itemId);
  imgElement.src = defaultPlaceholderImage;
  imgElement.classList.add("opacity-50");
  imageLoadingStates.value[itemId] = false;
};

// 添加图片加载处理函数
const handleImageLoad = (itemId: string) => {
  imageLoadingStates.value[itemId] = false;
};

// 添加图片加载开始处理函数
const handleImageLoadStart = (itemId: string) => {
  imageLoadingStates.value[itemId] = true;
};

// 修改模板以包含图片加载状态指示器
<template>
  <!-- ...existing code... -->
  <td class="whitespace-nowrap p-4 text-sm text-gray-500 border-r border-gray-100/50 text-center">
    <div class="relative inline-block">
      <img
        v-if="item.url || item.thumbnailUrl"
        ref="imageRefs"
        class="inline-block size-14 rounded-md object-cover cursor-pointer hover:opacity-90 transition-opacity shadow-sm hover:shadow-md"
        :src="failedImages.has(item.id) ? defaultPlaceholderImage : (item.thumbnailUrl ?? item.url)"
        :alt="item.name || DefaultPictureTexts.UNNAMED_PICTURE"
        @click="toggleFullscreen(item.id)"
        @error="(e) => handleImageError(e, item.id)"
        @load="() => handleImageLoad(item.id)"
        @loadstart="() => handleImageLoadStart(item.id)"
      />
      <div v-if="imageLoadingStates[item.id]"
           class="absolute inset-0 flex items中心 justify-center bg-gray-100/50 rounded-md">
        <div class="animate-spin rounded-full h-6 w-6 border-2 border-primary border-t-transparent"></div>
      </div>
    </div>
  </td>
  <!-- ...existing code... -->
</template>

// 修改 toggleFullscreen 函数以支持图片预加载
const toggleFullscreen = async (id: string | undefined) => {
  if (!id) return;

  const item = ListInfo.value.find((picture) => picture.id === id);
  if (!item?.url) return;

  try {
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.src = item.url!;
      img.onload = resolve;
      img.onerror = reject;
    });

    const imageElement = imageRefs.value.find(
      (ref) => ref?.getAttribute("src") === (item.thumbnailUrl ?? item.url)
    );

    if (imageElement) {
      imageElement.src = item.url;
      const { toggle } = useFullscreen(imageElement);
      toggle();

      imageElement.addEventListener("fullscreenchange", () => {
        if (!document.fullscreenElement && item.thumbnailUrl) {
          imageElement.src = item.thumbnailUrl;
        }
      }, { once: true });
    }
  } catch (error) {
    Message.error('Failed to load full size image');
  }
};

// 清理函数
onUnmounted(() => {
  imageRefs.value = [];
  failedImages.value.clear();
  imageLoadingStates.value = {};
});
```

这些修改解决了无限加载问题，改善了用户体验，并确保了更好的内存管理和错误处理。

![结论](/images/blog/bug/2025-2-12-success.png)

## 结论

通过实施这些优化，我们可以解决无限加载问题，并提升云图库应用中 Manager Pictures 页面的整体性能和可靠性。
