<script setup lang="ts">
interface ProjectItem {
  name?: string;
  link?: string;
  desc?: string;
  icon?: string;
}

interface Props {
  projects: Record<string, ProjectItem[]>;
}

const props = defineProps<Props>();

// 安全地获取项目数据，处理边界情况
const safeProjects = computed(() => {
  if (!props.projects || typeof props.projects !== 'object') {
    console.warn('ListProjects: projects prop is not a valid object');
    return {};
  }

  const result: Record<string, ProjectItem[]> = {};

  Object.entries(props.projects).forEach(([key, items]) => {
    if (key && Array.isArray(items)) {
      result[key] = items.filter(item =>
        item && typeof item === 'object' && item.name
      );
    }
  });

  return result;
});

// 安全地渲染HTML内容
const safeRenderDesc = (desc: string | undefined): string => {
  if (!desc || typeof desc !== 'string') return '';

  try {
    // 基本的XSS防护，移除潜在危险的标签
    return desc
      .replace(/<script[^>]*>.*?<\/script>/gi, '')
      .replace(/<iframe[^>]*>.*?<\/iframe>/gi, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '');
  } catch (error) {
    console.error('ListProjects: Error processing description:', error);
    return '';
  }
};

// 处理项目点击事件
const handleItemClick = (item: ProjectItem, event: Event) => {
  try {
    if (!item.link) {
      event.preventDefault();
      console.warn('ListProjects: No link provided for project:', item.name);
      return;
    }

    // 验证链接是否安全
    if (item.link.startsWith('javascript:') || item.link.includes('<script')) {
      event.preventDefault();
      console.error('ListProjects: Unsafe link detected:', item.link);
      return;
    }
  } catch (error) {
    event.preventDefault();
    console.error('ListProjects: Error handling click:', error);
  }
};
</script>

<template>
  <template v-for="(items, key) in safeProjects" :key="key">
    <h4 class="mt-10 font-bold">
      {{ key }}
    </h4>
    <div class="project-grid py-2 -mx-3 gap-2">
      <a v-for="(item, idx) in items" :key="`${key}-${idx}`" class="item relative flex items-center"
        :href="item.link || '#'" :target="item.link ? '_blank' : '_self'" :class="!item.link ? 'opacity-0 pointer-events-none h-0 -mt-8 -mb-4' : ''
          " @click="handleItemClick(item, $event)">
        <div v-if="item.icon" class="pt-2 pr-5">
          <Slidev v-if="item.icon === 'slidev'" class="text-4xl opacity-50" />
          <VueUse v-else-if="item.icon === 'vueuse'" class="text-4xl opacity-50" />
          <VueReactivity v-else-if="item.icon === 'vue-reactivity'" class="text-4xl opacity-50" />
          <VueDemi v-else-if="item.icon === 'vue-demi'" class="text-4xl opacity-50" />
          <Unocss v-else-if="item.icon === 'unocss'" class="text-4xl opacity-50" />
          <Vitest v-else-if="item.icon === 'vitest'" class="text-4xl opacity-50" />
          <div v-else class="text-3xl opacity-50" :class="item.icon || 'i-carbon-unknown'" />
        </div>
        <div class="flex-auto">
          <div class="text-normal">{{ item.name || 'Unnamed Project' }}</div>
          <div class="desc text-sm opacity-50 font-normal" v-html="safeRenderDesc(item.desc)" />
        </div>
      </a>
    </div>
  </template>
</template>

<style scoped>
.project-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.project-grid a.item {
  padding: 0.8em 1em;
  background: transparent;
  font-size: 1.1rem;
}

.project-grid a.item:hover {
  background: #88888808;
}
</style>
