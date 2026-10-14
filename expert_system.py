"""Reglas didácticas para convertir señales de una película en una recomendación."""

from __future__ import annotations


def evaluate_movie(
    predicted_rating: float, award_wins: int, award_nominations: int
) -> dict[str, str | list[str]]:
    """Aplica reglas IF/THEN transparentes; no modifica la nota estimada."""
    fired_rules: list[str] = []

    if predicted_rating >= 7.0:
        recommendation = "Recomendación favorable"
        fired_rules.append("R1: SI nota estimada >= 7, ENTONCES recomendación favorable.")
    elif predicted_rating >= 5.5:
        recommendation = "Opinión intermedia"
        fired_rules.append("R2: SI 5,5 <= nota estimada < 7, ENTONCES opinión intermedia.")
    else:
        recommendation = "Recomendación prudente"
        fired_rules.append("R3: SI nota estimada < 5,5, ENTONCES recomendación prudente.")

    has_award_signal = award_wins >= 3 or award_nominations >= 5
    if has_award_signal:
        fired_rules.append(
            "R4: SI hay al menos 3 premios o 5 nominaciones, ENTONCES existe señal de reconocimiento."
        )
    else:
        fired_rules.append(
            "R5: SI hay menos de 3 premios y menos de 5 nominaciones, "
            "ENTONCES no se activa la señal de reconocimiento."
        )

    if predicted_rating >= 7.0 and has_award_signal:
        conclusion = "La nota estimada y el reconocimiento apuntan en la misma dirección."
        fired_rules.append("R6: SI la nota es favorable y hay reconocimiento, ENTONCES señales convergentes.")
    elif predicted_rating < 5.5 and has_award_signal:
        conclusion = "Las señales discrepan: hay reconocimiento, pero la nota estimada es baja."
        fired_rules.append("R7: SI la nota es baja y hay reconocimiento, ENTONCES señalar discrepancia.")
    else:
        conclusion = "La recomendación se basa principalmente en la nota estimada."

    return {
        "recommendation": recommendation,
        "conclusion": conclusion,
        "fired_rules": fired_rules,
    }
