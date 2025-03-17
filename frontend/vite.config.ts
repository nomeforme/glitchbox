import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': 'http://192.168.1.247:7860',
      '/api/ws': {
        target: 'ws://192.168.1.247:7860',
        ws: true
      }
    }
  }
});

// http://100.79.41.86:7860