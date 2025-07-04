<template>
    <div class="mermaid-container">
        <div ref="mermaidRef" class="mermaid-diagram"></div>
    </div>
</template>

<script setup lang="ts">
import { isDark } from '~/logics'

const props = defineProps<{
    code: string
}>()

const mermaidRef = ref<HTMLDivElement>()

const renderMermaid = async () => {
    if (!mermaidRef.value) return
    
    // 在 SSR 环境中跳过渲染
    if (typeof window === 'undefined') return
    
    try {
        // 动态导入 mermaid
        const mermaidModule = await import('mermaid')
        const mermaid = mermaidModule.default
        
        mermaid.initialize({
            theme: isDark.value ? 'dark' : 'default',
            startOnLoad: false,
            fontFamily: 'inherit',
            themeVariables: {
                primaryColor: isDark.value ? '#FF9800' : '#3F51B5',
                primaryTextColor: isDark.value ? '#fff' : '#333',
                background: isDark.value ? '#1a1a1a' : '#ffffff'
            }
        })

        const { svg } = await mermaid.render('mermaid-' + Date.now(), props.code)
        mermaidRef.value.innerHTML = svg
    } catch (error) {
        console.error('Mermaid render error:', error)
        mermaidRef.value.innerHTML = `<pre class="mermaid-error">${props.code}</pre>`
    }
}

// 只在客户端挂载后执行
onMounted(() => {
    // 确保在客户端环境
    if (typeof window !== 'undefined') {
        // 使用 requestAnimationFrame 确保 DOM 完全准备好
        requestAnimationFrame(() => {
            renderMermaid()
        })
    }
})

// 监听主题变化
watch(isDark, () => {
    if (typeof window !== 'undefined') {
        renderMermaid()
    }
})
</script>

<style scoped>
.mermaid-container {
    margin: 1.5em 0;
    text-align: center;
}

.mermaid-diagram :deep(svg) {
    max-width: 100%;
    height: auto;
}

.mermaid-error {
    color: #cc0000;
    background: #f5f5f5;
    padding: 1em;
    border-radius: 4px;
    text-align: left;
}
</style>