import{_ as l}from"./Post.79f9855f.js";import{u as r,c,w as p,o as i,a,b as s}from"./app.c74dbdb6.js";var m="/assets/master.1919b4a1.png",k="/assets/success.af9b8fb4.png";const _="MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)",h="2025-03-01T00:00:00.000Z",f=!0,x="zh",y=[{property:"og:title",content:"MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)"}],q={__name:"mysql-sync",setup(u,{expose:t}){const n={title:"MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)",date:"2025-03-01T00:00:00.000Z",draft:!0,lang:"zh",meta:[{property:"og:title",content:"MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)"}]};return t({frontmatter:n}),r({title:"MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)",meta:[{property:"og:title",content:"MySQL\u4E3B\u4ECE\u6570\u636E\u5E93\u540C\u6B65(Docker\u90E8\u7F72)"}]}),(v,e)=>{const o=l;return i(),c(o,{frontmatter:n},{default:p(()=>e[0]||(e[0]=[a("div",{class:"prose m-auto"},[a("h1",{id:"\u90E8\u7F72\u4E3B\u6570\u636E\u5E93",tabindex:"-1"},[s("\u90E8\u7F72\u4E3B\u6570\u636E\u5E93 "),a("a",{class:"header-anchor",href:"#\u90E8\u7F72\u4E3B\u6570\u636E\u5E93","aria-hidden":"true"},"#")]),a("p",null,"\u4E3B\u6570\u636E\u5E93\u914D\u7F6E\u6587\u4EF6"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token punctuation"},"["),s("mysqld"),a("span",{class:"token punctuation"},"]"),s(`
datadir `),a("span",{class:"token operator"},"="),s(` /xxx/master1/data
`),a("span",{class:"token comment"},"# \u670D\u52A1\u5668\u552F\u4E00id\uFF0C\u9ED8\u8BA4\u503C1"),s(`
server-id`),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"1"),s(`
`),a("span",{class:"token comment"},"# \u8BBE\u7F6E\u65E5\u5FD7\u683C\u5F0F\uFF0C\u9ED8\u8BA4\u503CROW"),s(`
`),a("span",{class:"token assign-left variable"},"binlog_format"),a("span",{class:"token operator"},"="),s(`STATEMENT
`),a("span",{class:"token comment"},"# \u4E8C\u8FDB\u5236\u65E5\u5FD7\u540D\uFF0C\u9ED8\u8BA4binlog"),s(`
log-bin`),a("span",{class:"token operator"},"="),s(`binlog
`)])]),a("p",null,"docker \u90E8\u7F72\u4E3B\u6570\u636E\u5E93\u547D\u4EE4"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"docker"),s(" run "),a("span",{class:"token parameter variable"},"--name"),a("span",{class:"token operator"},"="),s("mysql-master-1 "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"--privileged"),a("span",{class:"token operator"},"="),s("true "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-p"),s(),a("span",{class:"token number"},"8808"),s(":3306 "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/master1/data/:/var/lib/mysql "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/master1/conf/my.cnf:/etc/mysql/my.cnf "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/master1/mysql-files/:/var/lib/mysql-files/ "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-e"),s(),a("span",{class:"token assign-left variable"},"MYSQL_ROOT_PASSWORD"),a("span",{class:"token operator"},"="),s("root "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-d"),s(" mysql:8.0 "),a("span",{class:"token parameter variable"},"--lower_case_table_names"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"1"),s(`
`)])]),a("h1",{id:"\u90E8\u7F72\u4ECE\u6570\u636E\u5E93",tabindex:"-1"},[s("\u90E8\u7F72\u4ECE\u6570\u636E\u5E93 "),a("a",{class:"header-anchor",href:"#\u90E8\u7F72\u4ECE\u6570\u636E\u5E93","aria-hidden":"true"},"#")]),a("p",null,"\u4ECE\u6570\u636E\u5E93\u914D\u7F6E\u6587\u4EF6"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token punctuation"},"["),s("mysqld"),a("span",{class:"token punctuation"},"]"),s(`
datadir `),a("span",{class:"token operator"},"="),s(` /xxx/slave1/data
server-id`),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"2"),s(`
`)])]),a("p",null,"docker \u90E8\u7F72\u4ECE\u6570\u636E\u5E93\u547D\u4EE4"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"docker"),s(" run "),a("span",{class:"token parameter variable"},"--name"),a("span",{class:"token operator"},"="),s("mysql-slave-1 "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"--privileged"),a("span",{class:"token operator"},"="),s("true "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-p"),s(),a("span",{class:"token number"},"8809"),s(":3306 "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/slave1/data/:/var/lib/mysql "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/slave1/conf/my.cnf:/etc/mysql/my.cnf "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-v"),s(" /xxx/slave1/mysql-files/:/var/lib/mysql-files/ "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-e"),s(),a("span",{class:"token assign-left variable"},"MYSQL_ROOT_PASSWORD"),a("span",{class:"token operator"},"="),s("root "),a("span",{class:"token punctuation"},"\\"),s(`
`),a("span",{class:"token parameter variable"},"-d"),s(" mysql:8.0 "),a("span",{class:"token parameter variable"},"--lower_case_table_names"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"1"),s(`
`)])]),a("h1",{id:"\u914D\u7F6E\u4E3B\u4ECE\u540C\u6B65",tabindex:"-1"},[s("\u914D\u7F6E\u4E3B\u4ECE\u540C\u6B65 "),a("a",{class:"header-anchor",href:"#\u914D\u7F6E\u4E3B\u4ECE\u540C\u6B65","aria-hidden":"true"},"#")]),a("p",null,"\u4E3B\u6570\u636E\u5E93"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"docker"),s(),a("span",{class:"token builtin class-name"},"exec"),s(),a("span",{class:"token parameter variable"},"-it"),s(` mysql-master-1 /bin/bash

mysql `),a("span",{class:"token parameter variable"},"-uroot"),s(),a("span",{class:"token parameter variable"},"-proot"),s(),a("span",{class:"token comment"},"# \u767B\u5F55"),s(`

CREATE `),a("span",{class:"token environment constant"},"USER"),s(),a("span",{class:"token string"},"'slave'"),s(" @"),a("span",{class:"token string"},"'%'"),s(" IDENTIFIED WITH mysql_native_password BY "),a("span",{class:"token string"},"'123456'"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u521B\u5EFA\u7528\u6237"),s(`

GRANT replication SLAVE ON*.*TO `),a("span",{class:"token string"},"'slave'"),s(" @"),a("span",{class:"token string"},"'%'"),a("span",{class:"token punctuation"},";"),s("  "),a("span",{class:"token comment"},"# \u6388\u6743"),s(`

flush privileges`),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u5237\u65B0\u6743\u9650"),s(`

show variables like `),a("span",{class:"token string"},"'server_id'"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u67E5\u770Bserver_id"),s(`

show master status`),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u67E5\u770B\u4E3B\u5E93\u7684binlog\u4FE1\u606F"),s(`
`)])]),a("p",null,[a("img",{src:m,alt:"\u4E3B\u6570\u636E\u5E93"})]),a("p",null,"\u4ECE\u6570\u636E\u5E93"),a("pre",{class:"language-bash"},[a("code",{class:"language-bash"},[a("span",{class:"token function"},"docker"),s(),a("span",{class:"token builtin class-name"},"exec"),s(),a("span",{class:"token parameter variable"},"-it"),s(` mysql-slave-1 /bin/bash

mysql `),a("span",{class:"token parameter variable"},"-uroot"),s(),a("span",{class:"token parameter variable"},"-proot"),s(),a("span",{class:"token comment"},"# \u767B\u5F55"),s(`

show variables like `),a("span",{class:"token string"},"'server_id'"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u67E5\u770Bserver_id"),s(`

`),a("span",{class:"token builtin class-name"},"set"),s(" global server_id "),a("span",{class:"token operator"},"="),s(),a("span",{class:"token number"},"2"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u8BBE\u7F6Eserver_id \u4ECE\u6570\u636E\u5E93\u9700\u8981\u4E0E\u4E3B\u6570\u636E\u5E93\u4E0D\u540C"),s(`

`),a("span",{class:"token comment"},"# \u82E5\u4E4B\u524D\u8BBE\u7F6E\u8FC7\u540C\u6B65\uFF0C\u8BF7\u5148\u91CD\u7F6E"),s(`
stop slave`),a("span",{class:"token punctuation"},";"),s(`
reset slave`),a("span",{class:"token punctuation"},";"),s(`

change master to `),a("span",{class:"token assign-left variable"},"master_host"),a("span",{class:"token operator"},"="),a("span",{class:"token string"},"'\u8FD9\u91CC\u4E0D\u80FD\u586B127.0.0.1'"),s(",master_port"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"8808"),s(",master_user"),a("span",{class:"token operator"},"="),a("span",{class:"token string"},"'slave'"),s(",master_password"),a("span",{class:"token operator"},"="),a("span",{class:"token string"},"'123456'"),s(",master_log_file"),a("span",{class:"token operator"},"="),a("span",{class:"token string"},"'binlog.000001'"),s(",master_log_pos"),a("span",{class:"token operator"},"="),a("span",{class:"token number"},"801"),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u8BBE\u7F6E\u4E3B\u6570\u636E\u5E93"),s(`
`),a("span",{class:"token comment"},"# \u6CE8\u610F:"),s(`
`),a("span",{class:"token comment"},"# master_log_file: \u4E3B\u6570\u636E\u5E93\u7684binlog\u6587\u4EF6\u540D(\u4E3B\u5E93\u6267\u884C show master status; \u83B7\u53D6 )"),s(`
`),a("span",{class:"token comment"},"# master_log_pos: \u4E3B\u6570\u636E\u5E93\u7684binlog\u6587\u4EF6\u4F4D\u7F6E(\u4E3B\u5E93\u6267\u884C show master status; \u83B7\u53D6 )"),s(`
`),a("span",{class:"token comment"},"# master_host: \u4E3B\u6570\u636E\u5E93\u7684ip\u5730\u5740 \u9700\u8981\u67E5\u770B\u4E3B\u673Aip\uFF0C\u4E0D\u80FD\u4F7F\u7528\u672C\u5730\u73AF\u56DE\u5730\u5740"),s(`
`),a("span",{class:"token comment"},"# master_port: \u4E3B\u6570\u636E\u5E93\u7684\u7AEF\u53E3\u53F7"),s(`

start slave`),a("span",{class:"token punctuation"},";"),s(`

show slave status`),a("span",{class:"token punctuation"},";"),s(),a("span",{class:"token comment"},"# \u67E5\u8BE2 Slave \u72B6\u6001"),s(`
`)])]),a("p",null,[a("img",{src:k,alt:"\u6210\u529F"})]),a("p",null,[a("strong",null,"\u6CE8\u610F: \u4ECE\u5E93\u8BBE\u7F6E\u65F6\uFF0C\u4E00\u5B9A\u4E0D\u80FD\u4F7F\u7528\u672C\u5730\u73AF\u56DE\u5730\u5740\uFF0C\u8FD9\u6837\u4F1A\u51FA\u73B0\u4E3B\u4ECE\u540C\u6B65\u5931\u8D25(Connecting \u72B6\u6001 \u662F\u4E2A\u5751)")])],-1)])),_:1})}}};export{h as date,q as default,f as draft,x as lang,y as meta,_ as title};
