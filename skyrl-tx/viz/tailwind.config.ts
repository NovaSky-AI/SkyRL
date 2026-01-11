import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "oklch(var(--background))",
        foreground: "oklch(var(--foreground))",
        card: {
          DEFAULT: "oklch(var(--card))",
          foreground: "oklch(var(--card-foreground))",
        },
        muted: {
          DEFAULT: "oklch(var(--muted))",
          foreground: "oklch(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "oklch(var(--accent))",
          foreground: "oklch(var(--accent-foreground))",
        },
        border: "oklch(var(--border))",
        ring: "oklch(var(--ring))",
        sidebar: {
          DEFAULT: "oklch(var(--sidebar))",
          foreground: "oklch(var(--sidebar-foreground))",
          accent: "oklch(var(--sidebar-accent))",
          "accent-foreground": "oklch(var(--sidebar-accent-foreground))",
          border: "oklch(var(--sidebar-border))",
        },
        "table-header": "oklch(var(--muted-foreground))",
        chart: {
          "1": "oklch(var(--chart-1))",
          "2": "oklch(var(--chart-2))",
          "3": "oklch(var(--chart-3))",
          "4": "oklch(var(--chart-4))",
          "5": "oklch(var(--chart-5))",
        },
        success: "oklch(var(--success))",
        warning: "oklch(var(--warning))",
        destructive: "oklch(var(--destructive))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "scale(0.98)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        "modal-bounce-in": {
          "0%": { opacity: "0", transform: "scale(0.9)" },
          "50%": { opacity: "1", transform: "scale(1.02)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        "modal-bounce-out": {
          "0%": { opacity: "1", transform: "scale(1)" },
          "100%": { opacity: "0", transform: "scale(0.95)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.3s ease",
        "modal-bounce-in": "modal-bounce-in 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) forwards",
        "modal-bounce-out": "modal-bounce-out 0.2s ease-out forwards",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
