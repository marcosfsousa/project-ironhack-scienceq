/// <reference types="vite/client" />

export {};

declare module "react" {
  interface HTMLAttributes<T> {
    inert?: boolean;
  }
}
