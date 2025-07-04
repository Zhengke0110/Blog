<template>
    <div class="mermaid-container">
        <ClientOnly>
            <div ref="mermaidRef" class="mermaid-diagram"></div>
            <template #fallback>
                <pre class="mermaid-loading">{{ code }}</pre>
            </template>
        </ClientOnly>
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

onMounted(() => {
    nextTick(() => {
        renderMermaid()
    })
})

watch(isDark, renderMermaid)
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

.mermaid-error,
.mermaid-loading {
    color: #666;
    background: #f5f5f5;
    padding: 1em;
    border-radius: 4px;
    text-align: left;
    font-family: monospace;
    font-size: 0.9em;
}

.mermaid-error {
    color: #cc0000;
}

.mermaid-loading {
    color: #666;
    border-left: 3px solid #ccc;
}
</style>