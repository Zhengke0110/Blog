<script setup lang="ts">
import { formatDate } from "~/logics";
import MermaidDiagram from './MermaidDiagram.vue'
import MathJax from './MathJax.vue'

// 声明 MathJax 类型
declare global {
  interface Window {
    MathJax: any
  }
}

defineProps({
  frontmatter: {
    type: Object,
    required: true,
  },
});

const router = useRouter();
const route = useRoute();
const content = ref<HTMLDivElement>();

onMounted(() => {
  const navigate = () => {
    if (location.hash) {
      document
        .querySelector(decodeURIComponent(location.hash))
        ?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleAnchors = (event: MouseEvent & { target: HTMLElement }) => {
    const link = event.target.closest("a");

    if (
      !event.defaultPrevented &&
      link &&
      event.button === 0 &&
      link.target !== "_blank" &&
      link.rel !== "external" &&
      !link.download &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.shiftKey &&
      !event.altKey
    ) {
      const url = new URL(link.href);
      if (url.origin !== window.location.origin) return;

      event.preventDefault();
      const { pathname, hash } = url;
      if (hash && (!pathname || pathname === location.pathname)) {
        window.history.replaceState({}, "", hash);
        navigate();
      } else {
        router.push({ path: pathname, hash });
      }
    }
  };

  // 处理 Mermaid 图表
  const processMermaidBlocks = () => {
    if (!content.value) return

    const mermaidBlocks = content.value.querySelectorAll('pre code.language-mermaid')
    mermaidBlocks.forEach((block) => {
      const code = block.textContent || ''
      const pre = block.parentElement
      if (pre) {
        const wrapper = document.createElement('div')
        wrapper.className = 'mermaid-wrapper'
        pre.parentNode?.replaceChild(wrapper, pre)

        // 使用 Vue 组件替换
        const app = createApp(MermaidDiagram, { code })
        app.mount(wrapper)
      }
    })
  }

  // 处理 MathJax 渲染
  const renderMathJax = async () => {
    await nextTick()
    // 检查 MathJax 是否可用
    if (typeof window !== 'undefined' && 
        window.MathJax && 
        window.MathJax.typesetPromise && 
        content.value) {
      try {
        await window.MathJax.typesetPromise([content.value])
        console.log('Post: MathJax rendering completed')
      } catch (err) {
        console.warn('Post: MathJax rendering error:', err)
      }
    }
  }

  useEventListener(window, "hashchange", navigate);
  useEventListener(content.value!, "click", handleAnchors, { passive: false });

  navigate();
  
  // 在导航逻辑后处理 Mermaid 和 MathJax
  nextTick(() => {
    processMermaidBlocks()
    renderMathJax()
  })
  
  setTimeout(navigate, 500);
});
</script>

<template>
  <div v-if="frontmatter.display ?? frontmatter.title" class="prose m-auto mb-8">
    <h1 class="mb-0">
      {{ frontmatter.display ?? frontmatter.title }}
    </h1>
    <p v-if="frontmatter.date" class="opacity-50 !-mt-2">
      {{ formatDate(frontmatter.date) }}
      <span v-if="frontmatter.duration">· {{ frontmatter.duration }}</span>
    </p>
    <p v-if="frontmatter.subtitle" class="opacity-50 !-mt-6 italic">
      {{ frontmatter.subtitle }}
    </p>
  </div>
  <article ref="content">
    <MathJax>
      <slot />
    </MathJax>
  </article>
  <div v-if="route.path !== '/'" class="prose m-auto mt-8 mb-8">
    <!-- <router-link
      :to="route.path.split('/').slice(0, -1).join('/') || '/'"
      class="font-mono no-underline opacity-50 hover:opacity-75"
    > -->
    <a @click="router.back()" class="font-mono no-underline opacity-50 hover:opacity-75"> cd ..</a>

    <!-- </router-link> -->
    <!--   :to="route.path.split('/').slice(0, -1).join('/') || '/'" -->
  </div>
</template>
