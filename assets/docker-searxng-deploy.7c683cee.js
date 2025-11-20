import{_ as r}from"./Post.dd5dd46a.js";import{u as p,c as l,w as c,o as k,a,b as s}from"./app.038d67a7.js";const g="\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898",b="2025-09-23T00:00:00.000Z",f="notes",v=[{property:"og:title",content:"\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898"}],x={__name:"docker-searxng-deploy",setup(i,{expose:t}){const n={title:"\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898",date:"2025-09-23T00:00:00.000Z",type:"notes",meta:[{property:"og:title",content:"\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898"}]};return t({frontmatter:n}),p({title:"\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898",meta:[{property:"og:title",content:"\u5173\u4E8ESearXNG\u65E0\u6CD5\u8FD4\u56DEJSON\u7684\u95EE\u9898"}]}),(d,e)=>{const o=r;return k(),l(o,{frontmatter:n},{default:c(()=>[...e[0]||(e[0]=[a("div",{class:"prose m-auto"},[a("h1",{id:"docker-\u90E8\u7F72-searxng",tabindex:"-1"},[s("Docker \u90E8\u7F72 SearXNG "),a("a",{class:"header-anchor",href:"#docker-\u90E8\u7F72-searxng","aria-hidden":"true"},"#")]),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"docker"),s(` pull searxng/searxng:latest

`),a("span",{class:"token function"},"docker"),s(" run "),a("span",{class:"token parameter variable"},"-p"),s(),a("span",{class:"token number"},"18080"),s(":8080 "),a("span",{class:"token punctuation"},"\\"),s(`
        `),a("span",{class:"token parameter variable"},"--name"),s(" searxng "),a("span",{class:"token punctuation"},"\\"),s(`
        `),a("span",{class:"token parameter variable"},"-d"),s(),a("span",{class:"token parameter variable"},"--restart"),a("span",{class:"token operator"},"="),s("always "),a("span",{class:"token punctuation"},"\\"),s(`
        `),a("span",{class:"token parameter variable"},"-v"),s(),a("span",{class:"token string"},'"\u7535\u8111\u4E0A\u6302\u8F7D\u7684\u5730\u5740/docker/SearXNG:/etc/searxng"'),s(),a("span",{class:"token punctuation"},"\\"),s(`
        `),a("span",{class:"token parameter variable"},"-e"),s(),a("span",{class:"token string"},[s('"BASE_URL=http://localhost:'),a("span",{class:"token variable"},"$PORT"),s('/"')]),s(),a("span",{class:"token punctuation"},"\\"),s(`
        `),a("span",{class:"token parameter variable"},"-e"),s(),a("span",{class:"token string"},'"INSTANCE_NAME=instance"'),s(),a("span",{class:"token punctuation"},"\\"),s(`
        searxng/searxng
`)])]),a("p",null,"\u6267\u884C\u641C\u7D22"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"curl"),s(),a("span",{class:"token parameter variable"},"-v"),s(),a("span",{class:"token string"},'"http://localhost:18080/search?q=\u4ECA\u5915\u662F\u4F55\u5E74&format=json"'),s(`
`)])]),a("p",null,"\u62A5\u9519\u5982\u4E0B"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[s(`

* Host localhost:18080 was resolved.
* IPv6: ::1
* IPv4: `),a("span",{class:"token number"},"127.0"),s(`.0.1
*   Trying `),a("span",{class:"token punctuation"},"["),s("::1"),a("span",{class:"token punctuation"},"]"),s(":18080"),a("span",{class:"token punctuation"},".."),s(`.
* Connected to localhost `),a("span",{class:"token punctuation"},"("),s("::1"),a("span",{class:"token punctuation"},")"),s(" port "),a("span",{class:"token number"},"18080"),s(`
`),a("span",{class:"token operator"},">"),s(" GET /search?q"),a("span",{class:"token operator"},"="),s("\u98CE\u95F4\u5F71\u6708"),a("span",{class:"token operator"},"&"),a("span",{class:"token assign-left variable"},"format"),a("span",{class:"token operator"},"="),s(`json HTTP/1.1
`),a("span",{class:"token operator"},">"),s(` Host: localhost:18080
`),a("span",{class:"token operator"},">"),s(` User-Agent: curl/8.7.1
`),a("span",{class:"token operator"},">"),s(` Accept: */*
`),a("span",{class:"token operator"},">"),s(`
* Request completely sent off
`),a("span",{class:"token operator"},"<"),s(" HTTP/1.1 "),a("span",{class:"token number"},"403"),s(` Forbidden
`),a("span",{class:"token operator"},"<"),s(" content-type: text/html"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token assign-left variable"},"charset"),a("span",{class:"token operator"},"="),s(`utf-8
`),a("span",{class:"token operator"},"<"),s(" content-length: "),a("span",{class:"token number"},"213"),s(`
`),a("span",{class:"token operator"},"<"),s(" server-timing: total"),a("span",{class:"token punctuation"},";"),a("span",{class:"token assign-left variable"},"dur"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"2.396"),s(", render"),a("span",{class:"token punctuation"},";"),a("span",{class:"token assign-left variable"},"dur"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"0"),s(`
`),a("span",{class:"token operator"},"<"),s(` x-content-type-options: nosniff
`),a("span",{class:"token operator"},"<"),s(` x-download-options: noopen
`),a("span",{class:"token operator"},"<"),s(` x-robots-tag: noindex, nofollow
`),a("span",{class:"token operator"},"<"),s(` referrer-policy: no-referrer
`),a("span",{class:"token operator"},"<"),s(` server: granian
`),a("span",{class:"token operator"},"<"),s(" date: Tue, "),a("span",{class:"token number"},"23"),s(" Sep "),a("span",{class:"token number"},"2025"),s(` 06:09:49 GMT
`),a("span",{class:"token operator"},"<"),s(`
`),a("span",{class:"token operator"},"<"),a("span",{class:"token operator"},"!"),s("doctype html"),a("span",{class:"token operator"},">"),s(`
`),a("span",{class:"token operator"},"<"),s("html "),a("span",{class:"token assign-left variable"},"lang"),a("span",{class:"token operator"},"="),s("en"),a("span",{class:"token operator"},">"),s(`
`),a("span",{class:"token operator"},"<"),s("title"),a("span",{class:"token operator"},">"),a("span",{class:"token number"},"403"),s(" Forbidden"),a("span",{class:"token operator"},"<"),s("/title"),a("span",{class:"token operator"},">"),s(`
`),a("span",{class:"token operator"},"<"),s("h"),a("span",{class:"token operator"},[a("span",{class:"token file-descriptor important"},"1"),s(">")]),s("Forbidden"),a("span",{class:"token operator"},"<"),s("/h"),a("span",{class:"token operator"},[a("span",{class:"token file-descriptor important"},"1"),s(">")]),s(`
`),a("span",{class:"token operator"},"<"),s("p"),a("span",{class:"token operator"},">"),s("You don"),a("span",{class:"token operator"},"&"),a("span",{class:"token comment"},"#39;t have the permission to access the requested resource. It is either read-protected or not readable by the server.</p>"),s(`
* Connection `),a("span",{class:"token comment"},"#0 to host localhost left intact"),s(`

`)])]),a("h1",{id:"\u89E3\u51B3\u65B9\u6848",tabindex:"-1"},[s("\u89E3\u51B3\u65B9\u6848 "),a("a",{class:"header-anchor",href:"#\u89E3\u51B3\u65B9\u6848","aria-hidden":"true"},"#")]),a("p",null,"\u5728\u914D\u7F6E\u6587\u4EF6\u4E2D\uFF0Cformats \u90E8\u5206\u53EA\u542F\u7528\u4E86 html \u683C\u5F0F\uFF0C\u9700\u8981\u989D\u5916\u542F\u7528 json"),a("ol",null,[a("li",null,[s("\u627E\u5230 "),a("code",null,"searxng/settings.yml"),s(" \u6587\u4EF6")]),a("li",null,[s("\u4FEE\u6539 "),a("code",null,"formats:"),s(" \u4E3A "),a("code",null,"formats: [html, json]")]),a("li",null,"\u91CD\u542F\u5BB9\u5668")])],-1)])]),_:1})}}};export{b as date,x as default,v as meta,g as title,f as type};
