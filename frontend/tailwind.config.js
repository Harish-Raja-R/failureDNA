/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        charcoal: {
          950: '#0B0F17',
          900: '#111726',
          800: '#1A2338',
        },
        slatepanel: {
          900: '#131B2E',
          800: '#1E293B',
          700: '#334155',
        },
        crimson: {
          500: '#EF4444',
          600: '#DC2626',
        },
        cyanaccent: {
          400: '#22D3EE',
          500: '#06B6D4',
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
