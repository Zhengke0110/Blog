import{_ as p}from"./Post.dd5dd46a.js";import{u as c,c as l,w as k,o as u,a as n,b as s}from"./app.038d67a7.js";const g="\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD",_="2025-08-13T00:00:00.000Z",w="notes",h=[{property:"og:title",content:"\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD"}],f={__name:"python-ollama-use",setup(i,{expose:o}){const a={title:"\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD",date:"2025-08-13T00:00:00.000Z",type:"notes",meta:[{property:"og:title",content:"\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD"}]};return o({frontmatter:a}),c({title:"\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD",meta:[{property:"og:title",content:"\u5982\u4F55\u5BF9 Ollama \u8FDB\u884C\u5355\u8F6E\u5BF9\u8BDD\u4E0E\u591A\u8F6E\u5BF9\u8BDD"}]}),(m,t)=>{const e=p;return u(),l(e,{frontmatter:a},{default:k(()=>[...t[0]||(t[0]=[n("div",{class:"prose m-auto"},[n("h1",{id:"\u5355\u8F6E\u5BF9\u8BDD\u5B9E\u73B0",tabindex:"-1"},[s("\u5355\u8F6E\u5BF9\u8BDD\u5B9E\u73B0 "),n("a",{class:"header-anchor",href:"#\u5355\u8F6E\u5BF9\u8BDD\u5B9E\u73B0","aria-hidden":"true"},"#")]),n("pre",{class:"language-python"},[n("code",{class:"language-python"},[n("span",{class:"token comment"},"# \u4F7F\u7528openai\u7684\u4EE3\u7801\u98CE\u683C\u8C03\u7528ollama"),s(`

`),n("span",{class:"token keyword"},"from"),s(" openai "),n("span",{class:"token keyword"},"import"),s(` OpenAI

`),n("span",{class:"token keyword"},"try"),n("span",{class:"token punctuation"},":"),s(`
    `),n("span",{class:"token comment"},"# \u521B\u5EFA\u5BA2\u6237\u7AEF"),s(`
    client `),n("span",{class:"token operator"},"="),s(" OpenAI"),n("span",{class:"token punctuation"},"("),s("base_url"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"http://localhost:11434/v1/"'),n("span",{class:"token punctuation"},","),s(" api_key"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"ollama"'),n("span",{class:"token punctuation"},")"),s(`

    `),n("span",{class:"token comment"},"# \u53D1\u9001\u8BF7\u6C42"),s(`
    chat_completion `),n("span",{class:"token operator"},"="),s(" client"),n("span",{class:"token punctuation"},"."),s("chat"),n("span",{class:"token punctuation"},"."),s("completions"),n("span",{class:"token punctuation"},"."),s("create"),n("span",{class:"token punctuation"},"("),s(`
        messages`),n("span",{class:"token operator"},"="),n("span",{class:"token punctuation"},"["),n("span",{class:"token punctuation"},"{"),n("span",{class:"token string"},'"role"'),n("span",{class:"token punctuation"},":"),s(),n("span",{class:"token string"},'"user"'),n("span",{class:"token punctuation"},","),s(),n("span",{class:"token string"},'"content"'),n("span",{class:"token punctuation"},":"),s(),n("span",{class:"token string"},'"\u4F60\u597D\uFF0C\u8BF7\u4ECB\u7ECD\u4E0B\u81EA\u5DF1"'),n("span",{class:"token punctuation"},"}"),n("span",{class:"token punctuation"},"]"),n("span",{class:"token punctuation"},","),s(" model"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"llama3"'),s(`
    `),n("span",{class:"token punctuation"},")"),s(`

    `),n("span",{class:"token comment"},"# \u8F93\u51FA\u7ED3\u679C"),s(`
    `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),s("chat_completion"),n("span",{class:"token punctuation"},"."),s("choices"),n("span",{class:"token punctuation"},"["),n("span",{class:"token number"},"0"),n("span",{class:"token punctuation"},"]"),n("span",{class:"token punctuation"},"."),s("message"),n("span",{class:"token punctuation"},"."),s("content"),n("span",{class:"token punctuation"},")"),s(`

`),n("span",{class:"token keyword"},"except"),s(" Exception "),n("span",{class:"token keyword"},"as"),s(" e"),n("span",{class:"token punctuation"},":"),s(`
    `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string-interpolation"},[n("span",{class:"token string"},'f"\u9519\u8BEF: '),n("span",{class:"token interpolation"},[n("span",{class:"token punctuation"},"{"),s("e"),n("span",{class:"token punctuation"},"}")]),n("span",{class:"token string"},'"')]),n("span",{class:"token punctuation"},")"),s(`
`)])]),n("h1",{id:"\u591A\u8F6E\u5BF9\u8BDD\u5B9E\u73B0",tabindex:"-1"},[s("\u591A\u8F6E\u5BF9\u8BDD\u5B9E\u73B0 "),n("a",{class:"header-anchor",href:"#\u591A\u8F6E\u5BF9\u8BDD\u5B9E\u73B0","aria-hidden":"true"},"#")]),n("pre",{class:"language-python"},[n("code",{class:"language-python"},[n("span",{class:"token comment"},"# \u591A\u8F6E\u5BF9\u8BDD"),s(`
`),n("span",{class:"token keyword"},"from"),s(" openai "),n("span",{class:"token keyword"},"import"),s(` OpenAI


`),n("span",{class:"token comment"},"# \u5B9A\u4E49\u591A\u8F6E\u5BF9\u8BDD\u65B9\u6CD5"),s(`
`),n("span",{class:"token keyword"},"def"),s(),n("span",{class:"token function"},"run_chat_session"),n("span",{class:"token punctuation"},"("),n("span",{class:"token punctuation"},")"),n("span",{class:"token punctuation"},":"),s(`
    `),n("span",{class:"token keyword"},"try"),n("span",{class:"token punctuation"},":"),s(`
        `),n("span",{class:"token comment"},"# \u521D\u59CB\u5316\u5BA2\u6237\u7AEF"),s(`
        client `),n("span",{class:"token operator"},"="),s(" OpenAI"),n("span",{class:"token punctuation"},"("),s("base_url"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"http://localhost:11434/v1/"'),n("span",{class:"token punctuation"},","),s(" api_key"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"ollama"'),n("span",{class:"token punctuation"},")"),s(`
        `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},`"\u804A\u5929\u5F00\u59CB\uFF01\u8F93\u5165 'exit' \u6216 'quit' \u9000\u51FA"`),n("span",{class:"token punctuation"},")"),s(`
        `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"-"'),s(),n("span",{class:"token operator"},"*"),s(),n("span",{class:"token number"},"40"),n("span",{class:"token punctuation"},")"),s(`

        `),n("span",{class:"token comment"},"# \u521D\u59CB\u5316\u5BF9\u8BDD\u5386\u53F2"),s(`
        chat_history `),n("span",{class:"token operator"},"="),s(),n("span",{class:"token punctuation"},"["),n("span",{class:"token punctuation"},"]"),s(`

        `),n("span",{class:"token comment"},"# \u542F\u52A8\u5BF9\u8BDD\u5FAA\u73AF"),s(`
        `),n("span",{class:"token keyword"},"while"),s(),n("span",{class:"token boolean"},"True"),n("span",{class:"token punctuation"},":"),s(`
            `),n("span",{class:"token keyword"},"try"),n("span",{class:"token punctuation"},":"),s(`
                `),n("span",{class:"token comment"},"# \u83B7\u53D6\u7528\u6237\u8F93\u5165"),s(`
                user_input `),n("span",{class:"token operator"},"="),s(),n("span",{class:"token builtin"},"input"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"\u7528\u6237\uFF1A"'),n("span",{class:"token punctuation"},")"),n("span",{class:"token punctuation"},"."),s("strip"),n("span",{class:"token punctuation"},"("),n("span",{class:"token punctuation"},")"),s(`
                `),n("span",{class:"token keyword"},"if"),s(" user_input"),n("span",{class:"token punctuation"},"."),s("lower"),n("span",{class:"token punctuation"},"("),n("span",{class:"token punctuation"},")"),s(),n("span",{class:"token keyword"},"in"),s(),n("span",{class:"token punctuation"},"["),n("span",{class:"token string"},'"exit"'),n("span",{class:"token punctuation"},","),s(),n("span",{class:"token string"},'"quit"'),n("span",{class:"token punctuation"},","),s(),n("span",{class:"token string"},'"\u9000\u51FA"'),n("span",{class:"token punctuation"},"]"),n("span",{class:"token punctuation"},":"),s(`
                    `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"\u9000\u51FA\u5BF9\u8BDD\u3002"'),n("span",{class:"token punctuation"},")"),s(`
                    `),n("span",{class:"token keyword"},"break"),s(`

                `),n("span",{class:"token keyword"},"if"),s(),n("span",{class:"token keyword"},"not"),s(" user_input"),n("span",{class:"token punctuation"},":"),s("  "),n("span",{class:"token comment"},"# \u5904\u7406\u7A7A\u8F93\u5165"),s(`
                    `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"\u8BF7\u8F93\u5165\u6709\u6548\u5185\u5BB9"'),n("span",{class:"token punctuation"},")"),s(`
                    `),n("span",{class:"token keyword"},"continue"),s(`

                `),n("span",{class:"token comment"},"# \u66F4\u65B0\u5BF9\u8BDD\u5386\u53F2(\u6DFB\u52A0\u7528\u6237\u8F93\u5165)"),s(`
                chat_history`),n("span",{class:"token punctuation"},"."),s("append"),n("span",{class:"token punctuation"},"("),n("span",{class:"token punctuation"},"{"),n("span",{class:"token string"},'"role"'),n("span",{class:"token punctuation"},":"),s(),n("span",{class:"token string"},'"user"'),n("span",{class:"token punctuation"},","),s(),n("span",{class:"token string"},'"content"'),n("span",{class:"token punctuation"},":"),s(" user_input"),n("span",{class:"token punctuation"},"}"),n("span",{class:"token punctuation"},")"),s(`

                `),n("span",{class:"token comment"},"# \u8C03\u7528\u6A21\u578B\u56DE\u7B54"),s(`
                chat_completion `),n("span",{class:"token operator"},"="),s(" client"),n("span",{class:"token punctuation"},"."),s("chat"),n("span",{class:"token punctuation"},"."),s("completions"),n("span",{class:"token punctuation"},"."),s("create"),n("span",{class:"token punctuation"},"("),s(`
                    messages`),n("span",{class:"token operator"},"="),s("chat_history"),n("span",{class:"token punctuation"},","),s(" model"),n("span",{class:"token operator"},"="),n("span",{class:"token string"},'"llama3"'),s(`
                `),n("span",{class:"token punctuation"},")"),s(`

                `),n("span",{class:"token comment"},"# \u83B7\u53D6\u6700\u65B0\u56DE\u7B54"),s(`
                model_response `),n("span",{class:"token operator"},"="),s(" chat_completion"),n("span",{class:"token punctuation"},"."),s("choices"),n("span",{class:"token punctuation"},"["),n("span",{class:"token number"},"0"),n("span",{class:"token punctuation"},"]"),s(`
                `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"AI:"'),n("span",{class:"token punctuation"},","),s(" model_response"),n("span",{class:"token punctuation"},"."),s("message"),n("span",{class:"token punctuation"},"."),s("content"),n("span",{class:"token punctuation"},")"),s(`
                `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"-"'),s(),n("span",{class:"token operator"},"*"),s(),n("span",{class:"token number"},"40"),n("span",{class:"token punctuation"},")"),s(`

                `),n("span",{class:"token comment"},"# \u66F4\u65B0\u5BF9\u8BDD\u5386\u53F2\uFF08\u6DFB\u52A0AI\u6A21\u578B\u7684\u56DE\u590D\uFF09"),s(`
                chat_history`),n("span",{class:"token punctuation"},"."),s("append"),n("span",{class:"token punctuation"},"("),s(`
                    `),n("span",{class:"token punctuation"},"{"),n("span",{class:"token string"},'"role"'),n("span",{class:"token punctuation"},":"),s(),n("span",{class:"token string"},'"assistant"'),n("span",{class:"token punctuation"},","),s(),n("span",{class:"token string"},'"content"'),n("span",{class:"token punctuation"},":"),s(" model_response"),n("span",{class:"token punctuation"},"."),s("message"),n("span",{class:"token punctuation"},"."),s("content"),n("span",{class:"token punctuation"},"}"),s(`
                `),n("span",{class:"token punctuation"},")"),s(`

            `),n("span",{class:"token keyword"},"except"),s(" KeyboardInterrupt"),n("span",{class:"token punctuation"},":"),s(`
                `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},'"\\n\u7528\u6237\u4E2D\u65AD\u5BF9\u8BDD"'),n("span",{class:"token punctuation"},")"),s(`
                `),n("span",{class:"token keyword"},"break"),s(`
            `),n("span",{class:"token keyword"},"except"),s(" Exception "),n("span",{class:"token keyword"},"as"),s(" e"),n("span",{class:"token punctuation"},":"),s(`
                `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string-interpolation"},[n("span",{class:"token string"},'f"\u5BF9\u8BDD\u9519\u8BEF\uFF1A'),n("span",{class:"token interpolation"},[n("span",{class:"token punctuation"},"{"),s("e"),n("span",{class:"token punctuation"},"}")]),n("span",{class:"token string"},'"')]),n("span",{class:"token punctuation"},")"),s(`
                `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string"},`"\u7EE7\u7EED\u5BF9\u8BDD\uFF0C\u8F93\u5165 'exit' \u9000\u51FA"`),n("span",{class:"token punctuation"},")"),s(`
                `),n("span",{class:"token keyword"},"continue"),s(`

    `),n("span",{class:"token keyword"},"except"),s(" Exception "),n("span",{class:"token keyword"},"as"),s(" e"),n("span",{class:"token punctuation"},":"),s(`
        `),n("span",{class:"token keyword"},"print"),n("span",{class:"token punctuation"},"("),n("span",{class:"token string-interpolation"},[n("span",{class:"token string"},'f"\u521D\u59CB\u5316\u9519\u8BEF: '),n("span",{class:"token interpolation"},[n("span",{class:"token punctuation"},"{"),s("e"),n("span",{class:"token punctuation"},"}")]),n("span",{class:"token string"},'"')]),n("span",{class:"token punctuation"},")"),s(`

`),n("span",{class:"token keyword"},"if"),s(" __name__ "),n("span",{class:"token operator"},"=="),s(),n("span",{class:"token string"},'"__main__"'),n("span",{class:"token punctuation"},":"),s(`
    run_chat_session`),n("span",{class:"token punctuation"},"("),n("span",{class:"token punctuation"},")"),s(`
`)])])],-1)])]),_:1})}}};export{_ as date,f as default,h as meta,g as title,w as type};
