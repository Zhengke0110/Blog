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

    const { default: mermaid } = await import('mermaid')

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

    try {
        const { svg } = await mermaid.render('mermaid-' + Date.now(), props.code)
        mermaidRef.value.innerHTML = svg
    } catch (error) {
        console.error('Mermaid render error:', error)
        mermaidRef.value.innerHTML = `<pre class="mermaid-error">${props.code}</pre>`
    }
}

onMounted(renderMermaid)
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

.mermaid-error {
    color: #cc0000;
    background: #f5f5f5;
    padding: 1em;
    border-radius: 4px;
    text-align: left;
}
</style>