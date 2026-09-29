import type { CSSProperties } from "react";

/**
 * Cadre au format exact de la vidéo, pas plus haut que l'écran moins `reserve` px (en-tête, titre, commandes) :
 * une vidéo verticale se voit en entier sans défiler. Le format exact garde alignés les calques posés dessus
 * (rideau avant / après, boîtes des visages, zone à sélectionner).
 */
export function fitToScreen(width: number, height: number, reserve = 250): CSSProperties {
  return {
    aspectRatio: `${width} / ${height}`,
    width: `min(100%, max(320px, 100vh - ${reserve}px) * ${(width / height).toFixed(4)})`,
    marginInline: "auto",
  };
}
