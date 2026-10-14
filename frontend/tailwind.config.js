/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        soc: {
          bg: '#070a13',
          card: '#0d1322',
          surface: '#121a2f',
          elevated: '#18233d',
          border: 'rgba(56, 189, 248, 0.12)',
          borderHover: 'rgba(56, 189, 248, 0.3)',
          cyan: '#00f2fe',
          cyanGlow: 'rgba(0, 242, 254, 0.25)',
          violet: '#8b5cf6',
          violetGlow: 'rgba(139, 92, 246, 0.25)',
          danger: '#ef4444',
          warning: '#f59e0b',
          success: '#10b981',
          muted: '#94a3b8',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 20px -3px rgba(0, 242, 254, 0.3)',
        'glow-violet': '0 0 20px -3px rgba(139, 92, 246, 0.3)',
        'glow-danger': '0 0 20px -3px rgba(239, 68, 68, 0.35)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scanline': 'scanline 8s linear infinite',
      },
      keyframes: {
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(1000%)' },
        }
      }
    },
  },
  plugins: [],
}
