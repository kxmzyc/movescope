/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    // globals 供 @testing-library/react 注册 afterEach 自动清理 DOM。
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
