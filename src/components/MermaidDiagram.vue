<template>
    <div class="mermaid-container">
        <div ref="mermaidRef" class="mermaid-diagram"></div>
    </div>
</template>

<script setup lang="ts">
import { isDark } from '~/logics'

declare global {
    interface Window {
        mermaid: any
    }
}

const props = defineProps<{
    code: string
}>()

const mermaidRef = ref<HTMLDivElement>()
const isLoading = ref(true)
const loadError = ref(false)

// 确保 Mermaid 可用
const ensureMermaidAvailable = (): Promise<void> => {
    return new Promise((resolve, reject) => {
        if (window.mermaid) {
            resolve()
            return
        }
        
        // 等待一小段时间以确保脚本加载完成
        const checkMermaid = () => {
            if (window.mermaid) {
                resolve()
            } else {
                setTimeout(checkMermaid, 100)
            }
        }
        
        // 设置超时以防脚本加载失败
        setTimeout(() => {
            if (!window.mermaid) {
                reject(new Error('Mermaid script failed to load'))
            }
        }, 5000)
        
        checkMermaid()
    })
}

const renderMermaid = async () => {
    if (!mermaidRef.value) return

    try {
        isLoading.value = true
        loadError.value = false

        // 确保 Mermaid 脚本已加载
        await ensureMermaidAvailable()

        // 配置 Mermaid
        window.mermaid.initialize({
            theme: isDark.value ? 'dark' : 'default',
            startOnLoad: false,
            fontFamily: 'inherit',
            themeVariables: {
                primaryColor: isDark.value ? '#FF9800' : '#3F51B5',
                primaryTextColor: isDark.value ? '#fff' : '#333',
                background: isDark.value ? '#1a1a1a' : '#ffffff'
            }
        })

        // 渲染图表
        const { svg } = await window.mermaid.render('mermaid-' + Date.now(), props.code)
        mermaidRef.value.innerHTML = svg

    } catch (error) {
        console.error('Mermaid render error:', error)
        loadError.value = true
        if (mermaidRef.value) {
            mermaidRef.value.innerHTML = `<pre class="mermaid-error">${props.code}</pre>`
        }
    } finally {
        isLoading.value = false
    }
}

onMounted(() => {
    nextTick(() => {
        renderMermaid()
    })
})

watch(isDark, () => {
    if (!isLoading.value && !loadError.value) {
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