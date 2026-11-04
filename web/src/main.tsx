import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// Dev only: ?theme=dark|light forces a theme (screenshots).
const theme = import.meta.env.DEV ? new URLSearchParams(location.search).get("theme") : null;
if (theme) document.documentElement.dataset.theme = theme;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
