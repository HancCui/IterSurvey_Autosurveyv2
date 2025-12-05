---
config:
  theme: base
  themeVariables:
    lineColor: '#F8B229'
    primaryBorderColor: '#7C0000'
    primaryColor: '#BB2528'
    primaryTextColor: '#fff'
    secondaryColor: '#006100'
    tertiaryColor: '#fff'
---
graph TD
    A[开始] --> B[处理数据]
    B --> C[分析结果]
    C --> D[结束]
    
    classDef default fill:#BB2528,stroke:#7C0000,color:#fff;
