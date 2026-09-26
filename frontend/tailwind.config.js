/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          primary: "#0F766E",       // Deep clinical teal (NICE/NHS style)
          "primary-hover": "#115E59",
          "primary-light": "#CCFBF1",
          secondary: "#1E293B",     // Navy slate
          surface: "#F8FAFC",       // Clean light background
          "surface-card": "#FFFFFF",
          "surface-dark": "#0B0F17",  // High-contrast viewer canvas background
          border: "#E2E8F0",
          "border-dark": "#1E293B"
        },
        status: {
          pass: "#059669",          // Accessible Green (contrast > 4.5:1)
          "pass-bg": "#ECFDF5",
          fail: "#DC2626",          // Accessible Red
          "fail-bg": "#FEF2F2",
          warning: "#D97706",       // Accessible Amber
          "warning-bg": "#FFFBEB",
          info: "#2563EB",          // High-visibility Blue
          "info-bg": "#EFF6FF"
        },
        "dr-severity": {
          grade0: "#10B981",        // No DR (Muted Green)
          grade1: "#06B6D4",        // Mild NPDR (Cyan)
          grade2: "#F59E0B",        // Moderate NPDR (Amber)
          grade3: "#F97316",        // Severe NPDR (Orange)
          grade4: "#EF4444"         // Proliferative DR (Crimson)
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"]
      }
    },
  },
  plugins: [],
}
