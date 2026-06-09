---
name: react-patterns
description: React 组件规范：Hooks 用法、Props 类型、性能优化、项目约定
size: small
triggers: [React, component, hooks, useState, useEffect, props, TypeScript, JSX]
---

## React 组件规范

### 基本规则
- 使用函数式组件，禁止 class component
- 所有 Props 必须有 TypeScript 接口定义
- 组件文件名与组件名一致，使用 PascalCase

### 组件结构模板
```tsx
interface Props {
  // 必填 props 不加 ?，可选 props 加 ?
  title: string;
  onClose?: () => void;
}

export function MyComponent({ title, onClose }: Props) {
  // 1. Hooks（顺序固定：state → ref → context → effect）
  const [count, setCount] = useState(0);

  // 2. 派生值（不需要 state 的计算）
  const doubled = count * 2;

  // 3. 事件处理
  const handleClick = () => setCount(c => c + 1);

  // 4. Effects（最后）
  useEffect(() => {
    // cleanup
    return () => {};
  }, []);

  return <div>{title}</div>;
}
```

### Hooks 规范
- `useState`：状态更新用函数式写法 `setState(prev => prev + 1)`
- `useEffect`：依赖数组必须完整，不能省略
- 自定义 Hook 必须以 `use` 开头，抽取复用逻辑

### 性能优化
- 只在性能问题出现后才用 `memo`/`useMemo`/`useCallback`，不提前优化
- 列表渲染必须有稳定的 `key`（不用 index，除非列表不变）

### 样式
- 使用 CSS Modules（`styles.module.css`），不用内联样式
- 类名用 camelCase：`styles.cardTitle`

### 目录结构
```
components/
  UserCard/
    UserCard.tsx
    UserCard.module.css
    index.ts          ← 只 re-export: export { UserCard }
```
