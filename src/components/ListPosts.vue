<script setup lang="ts">
import { useRouter } from "vue-router";
import { formatDate } from "~/logics";

export interface Post {
  path: string;
  title: string;
  date: string;
  lang?: string;
  duration?: string;
}

const props = defineProps<{
  type?: string;
  posts?: Post[];
  sortBy?: 'date' | 'filename';
}>();

const router = useRouter();
const routes = router
  .getRoutes()
  .filter((i) => i.path.startsWith("/posts") && i.meta.frontmatter.date);

// 根据 sortBy 参数决定排序方式
const sortedRoutes = computed(() => {
  const filtered = routes.filter(
    (i) =>
      !i.path.endsWith(".html") && i.meta.frontmatter.type === props.type
  );

  if (props.sortBy === 'filename') {
    // 按文件名排序
    return filtered.sort((a, b) => {
      const aFilename = a.path.split('/').pop() || '';
      const bFilename = b.path.split('/').pop() || '';
      return aFilename.localeCompare(bFilename);
    });
  } else {
    // 默认按日期排序（最新的在前）
    return filtered.sort(
      (a, b) =>
        +new Date(b.meta.frontmatter.date) - +new Date(a.meta.frontmatter.date)
    );
  }
});

const posts = computed(
  () =>
    props.posts ||
    sortedRoutes.value.map((i) => ({
      path: i.path,
      title: i.meta.frontmatter.title,
      date: i.meta.frontmatter.date,
      lang: i.meta.frontmatter.lang,
      duration: i.meta.frontmatter.duration,
    }))
);
</script>

<template>
  <ul>
    <app-link v-for="route in posts" :key="route.path" class="item block font-normal mb-6 mt-2 no-underline"
      :to="route.path">
      <li class="no-underline">
        <div class="title text-lg">
          {{ route.title }}
          <sup v-if="route.lang === 'zh'" class="text-xs border border-current rounded px-1 pb-0.2">中文</sup>
        </div>
        <div class="time opacity-50 text-sm -mt-1">
          {{ formatDate(route.date) }}
          <span v-if="route.duration" class="opacity-50">· {{ route.duration }}</span>
        </div>
      </li>
    </app-link>
  </ul>
</template>
