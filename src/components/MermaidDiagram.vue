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

// 加载本地 Mermaid 文件
const loadMermaidScript = (): Promise<void> => {
    return new Promise((resolve, reject) => {
        // 如果已经加载过，直接返回
        if (window.mermaid) {
            resolve()
            return
        }

        // 检查是否已经有脚本在加载
        const existingScript = document.querySelector('script[src="/lib/mermaid.min.js"]')
        if (existingScript) {
            existingScript.addEventListener('load', () => resolve())
            existingScript.addEventListener('error', () => reject(new Error('Failed to load mermaid script')))
            return
        }

        // 创建并加载脚本
        const script = document.createElement('script')
        script.src = '/lib/mermaid.min.js'
        script.type = 'text/javascript'

        script.onload = () => {
            if (window.mermaid) {
                resolve()
            } else {
                reject(new Error('Mermaid not available after loading'))
            }
        }

        script.onerror = () => {
            reject(new Error('Failed to load mermaid script'))
        }

        document.head.appendChild(script)
    })
}

const renderMermaid = async () => {
    if (!mermaidRef.value) return

    try {
        isLoading.value = true
        loadError.value = false

        // 确保 Mermaid 脚本已加载
        await loadMermaidScript()

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