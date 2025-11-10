"use client"

import { forwardRef } from "react"
import { cn } from "@/lib/utils"

export const Switch = forwardRef(function Switch(
  { className, checked = false, onCheckedChange, disabled = false, ...props },
  ref
) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      ref={ref}
      onClick={() => {
        if (!disabled && typeof onCheckedChange === "function") {
          onCheckedChange(!checked)
        }
      }}
      className={cn(
        "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
        checked ? "bg-primary" : "bg-muted",
        className
      )}
      {...props}
    >
      <span
        className={cn(
          "inline-block h-5 w-5 transform rounded-full bg-background shadow transition-transform",
          checked ? "translate-x-5" : "translate-x-1"
        )}
      />
    </button>
  )
})

