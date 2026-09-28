import { Loader2 } from "lucide-react";
import { useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from "react";

/** Petit menu déroulant : se ferme au clic extérieur et sur Échap. À placer dans un parent `relative`. */
export function Menu({ open, onClose, children, align = "left", className }: { open: boolean; onClose: () => void; children: ReactNode; align?: "left" | "right"; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !ref.current?.parentElement?.contains(e.target as Node) && onClose();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      ref={ref}
      role="menu"
      className={cx(
        "absolute top-full z-50 mt-1.5 min-w-52 rounded-xl bg-overlay p-1.5 shadow-[var(--shadow-float)] ring-1 ring-line-strong",
        align === "left" ? "left-0" : "right-0",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function MenuItem({ onClick, children, active }: { onClick: () => void; children: ReactNode; active?: boolean }) {
  return (
    <button
      role="menuitem"
      onClick={onClick}
      className={cx("flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors hover:bg-raised", active && "bg-raised")}
    >
      {children}
    </button>
  );
}

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
};

export function Button({ variant = "secondary", size = "md", loading, icon, className, children, disabled, ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium whitespace-nowrap transition-[background,color,box-shadow,transform] duration-150 active:scale-[0.98] disabled:opacity-40 disabled:active:scale-100",
        size === "sm" && "h-8 px-3 text-[13px]",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-[15px]",
        variant === "primary" && "bg-accent text-accent-ink hover:bg-[#d6ff6b] shadow-[0_0_0_1px_rgb(200_255_61/0.4),0_8px_24px_-8px_rgb(200_255_61/0.45)]",
        variant === "secondary" && "bg-raised text-fg ring-1 ring-line hover:bg-overlay hover:ring-line-strong",
        variant === "ghost" && "text-muted hover:text-fg hover:bg-raised",
        variant === "danger" && "bg-danger-soft text-danger ring-1 ring-danger/30 hover:bg-danger/20",
        className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cx("rounded-[var(--radius-card)] bg-surface ring-1 ring-line", className)}>{children}</div>;
}

export function SegmentedControl<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode; icon?: ReactNode }[];
  className?: string;
}) {
  return (
    <div role="radiogroup" className={cx("inline-flex rounded-xl bg-bg p-1 ring-1 ring-line", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "inline-flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors",
            value === o.value ? "bg-overlay text-fg shadow-sm ring-1 ring-line-strong" : "text-muted hover:text-fg",
          )}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cx("flex items-start justify-between gap-4 py-1", disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer")}>
      <span>
        <span className="block text-sm font-medium">{label}</span>
        {description && <span className="mt-0.5 block text-[13px] text-muted">{description}</span>}
      </span>
      <button
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          "relative mt-0.5 h-6 w-10 shrink-0 rounded-full transition-colors",
          checked ? "bg-accent" : "bg-overlay ring-1 ring-line-strong",
          disabled && "pointer-events-none",
        )}
      >
        <span
          className={cx(
            "absolute top-0.5 left-0.5 size-5 rounded-full shadow transition-transform",
            checked ? "translate-x-4 bg-accent-ink" : "bg-muted",
          )}
        />
      </button>
    </label>
  );
}

export function ProgressBar({ value, tone = "accent", className }: { value: number; tone?: "accent" | "warn"; className?: string }) {
  return (
    <div className={cx("h-1.5 overflow-hidden rounded-full bg-overlay", className)}>
      <div
        className={cx("h-full rounded-full transition-[width] duration-500 ease-out", tone === "accent" ? "bg-accent" : "bg-warn")}
        style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }}
      />
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-overlay px-1.5 font-mono text-[11px] text-fg ring-1 ring-line-strong">
      {children}
    </kbd>
  );
}

export function Notice({ tone = "warn", children, className }: { tone?: "warn" | "danger" | "info"; children: ReactNode; className?: string }) {
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={cx(
        "rounded-xl px-4 py-3 text-sm",
        tone === "warn" && "bg-warn-soft text-warn ring-1 ring-warn/25",
        tone === "danger" && "bg-danger-soft text-danger ring-1 ring-danger/25",
        tone === "info" && "bg-raised text-muted ring-1 ring-line",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function SectionTitle({ eyebrow, title, subtitle, right }: { eyebrow?: string; title: ReactNode; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && <div className="mb-2 font-mono text-[11px] tracking-[0.18em] text-accent uppercase">{eyebrow}</div>}
        <h1 className="text-2xl font-semibold tracking-tight text-balance sm:text-[28px]">{title}</h1>
        {subtitle && <p className="mt-1.5 max-w-2xl text-[15px] text-muted text-pretty">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}
