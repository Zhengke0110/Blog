<template>
    <div ref="mathJaxContainer" class="mathjax-container">
        <slot />
    </div>
</template>

<script setup lang="ts">
import { onMounted, onUpdated, ref, nextTick } from 'vue'

const mathJaxContainer = ref<HTMLElement>()

// 简化版本：主要依赖 markdown-it-mathjax3 在构建时处理公式
const processMathElements = () => {
    if (!mathJaxContainer.value) return

    try {
        // 查找 MathJax 相关元素
        const mathElements = mathJaxContainer.value.querySelectorAll('mjx-container, svg[data-mml-node], .MathJax, [class*="mjx"]')

        mathElements.forEach((element, index) => {
            if (!element.classList.contains('mathjax-processed')) {
                element.classList.add('mathjax-processed')
            }
        })

        // 查找任何包含数学公式的文本节点
        const walker = document.createTreeWalker(
            mathJaxContainer.value,
            NodeFilter.SHOW_TEXT,
            null
        )

        let node
        while (node = walker.nextNode()) {
            if (node.textContent && (node.textContent.includes('$') || node.textContent.includes('\\('))) {
                console.log('Found potential math text:', node.textContent.substring(0, 50))
            }
        }

    } catch (error) {
        console.warn('Error processing math elements:', error)
    }
}

onMounted(() => {
    nextTick(() => {
        processMathElements()
    })
})

onUpdated(() => {
    nextTick(() => {
        processMathElements()
    })
})
</script>

<style scoped>
.mathjax-container {
    /* 确保数学公式继承文档的字体和颜色 */
    color: inherit;
    font-family: inherit;
    line-height: inherit;
}
</style>
