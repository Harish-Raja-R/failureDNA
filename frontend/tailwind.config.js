/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          50: '#FCFCFA',
          100: '#F8F8F5',
          200: '#F1EFEA',
        },
        charcoal: {
          950: '#0F0F0F',
          900: '#171717',
          800: '#262626',
          700: '#404040',
        },
        graphite: {
          500: '#737373',
          400: '#A3A3A3',
          300: '#D4D4D4',
        },
        brand: {
          pink: '#F0448A',
          pinkHover: '#D83275',
          pinkLight: '#FDE8F1',
        },
        emerald: {
          DEFAULT: '#15966A',
          hover: '#0E7A55',
          light: '#E7F7F1',
        },
        lavender: {
          soft: '#EAE5FF',
          border: '#D9D0FF',
        },
        border: {
          warm: '#E7E5E0',
          warmDark: '#D8D5CE',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      boxShadow: {
        subtle: '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02)',
        card: '0 4px 20px -2px rgba(23, 23, 23, 0.04), 0 2px 6px -1px rgba(23, 23, 23, 0.02)',
        hover: '0 10px 30px -4px rgba(23, 23, 23, 0.08), 0 4px 10px -2px rgba(23, 23, 23, 0.03)',
        spotlight: '0 0 40px -10px rgba(240, 68, 138, 0.15)',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
