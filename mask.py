import sys
import tensorflow as tf
import numpy as np
import transformers

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM, tokenization_utils_base

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Gibt den Index des ersten [MASK]-Tokens zurück oder None, falls kein [MASK] vorhanden ist.
    """
    token_ids = inputs['input_ids'].numpy().tolist()[0]  # Konvertiere zu einer Liste
    for i, token_id in enumerate(token_ids):
        if token_id == mask_token_id:
            return i  # Gib sofort den ersten gefundenen Index zurück

    return None  # Falls kein [MASK] gefunden wurde


def get_color_for_attention_score(attention_score):
    """
    Wandelt einen Attention-Score (0 bis 1) in eine Graustufenfarbe (0 bis 255) um.
    Höhere Attention-Werte ergeben hellere Farben.
    """
    # Begrenze den Wert auf den Bereich [0,1]
    attention_score = max(0.0, min(1.0, float(attention_score)))
    
    # Berechne die Intensität umgekehrt (höhere Werte = heller)
    intensity = int((1 - attention_score) * 255)
    
    return (intensity, intensity, intensity)

import csv

def save_attention_data(attention_data):
    """
    Speichert die wichtigsten Tokens mit der höchsten Aufmerksamkeit in einer CSV-Datei.
    """
    with open("attention_analysis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Layer", "Head", "Top Attended Tokens"])
        for row in attention_data:
            writer.writerow(row)
    
def visualize_attentions(inputs, attentions):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 🔍 Sicherstellen, dass inputs richtig verarbeitet wird
    # Falls `inputs` eine Liste ist, konvertiere es zurück in ein BatchEncoding
    if isinstance(inputs, list):
        inputs = tokenizer(" ".join(inputs), return_tensors="tf")

    # Falls `inputs` ein BatchEncoding-Objekt ist, wandle es in ein Dictionary um
    if isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
        inputs = dict(inputs)

        # Stelle sicher, dass `input_ids` existiert
    if "input_ids" in inputs:
        input_ids = inputs["input_ids"]
    else:
        raise ValueError(f"❌ Fehler: `inputs` enthält kein 'input_ids'-Feld! Struktur: {inputs}")

    # Falls `input_ids` ein Tensor ist, konvertiere ihn in eine Liste von Token-IDs
    if isinstance(input_ids, tf.Tensor):
        input_ids = input_ids.numpy().tolist()[0]

    # Konvertiere Token-IDs in die tatsächlichen Wörter
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Falls `attentions` weniger Tokens enthält als `tokens`, passe `tokens` an.
    min_dim = attentions[0][0].shape[0]  # Verwende die echte Größe von `attentions`
    if len(tokens) > min_dim:
        print(f"⚠️ Kürze Tokens von {len(tokens)} auf {min_dim}, da `attentions` kleiner ist.")
        tokens = tokens[:min_dim]

    # Sicherstellen, dass die Anzahl der Tokens mit den Dimensionen der Attention-Gewichte übereinstimmt
    min_dim = min(attentions[0].shape[-1], len(tokens))
    tokens = tokens[:min_dim]  # Tokens auf die minimale Dimension kürzen

    for layer_idx, layer in enumerate(attentions):
        for head_idx, head in enumerate(layer):
            attention_weights = head[0][:min_dim, :min_dim]  # Attention-Gewichte zuschneiden
            generate_diagram(layer_idx + 1, head_idx + 1, tokens, attention_weights)

            # Debugging: Zeige die neue angepasste Dimension
            print(f"✅ Anpassung für Layer {layer_idx+1}, Head {head_idx+1}: {attention_weights.shape}")
            print(f"🔍 `attentions` Shape: {attentions[0].shape}, `tokens` Count: {len(tokens)}")

    # Sicherstellen, dass Tokens nicht leer sind
    if not tokens:
        raise ValueError("❌ Fehler: `tokens` ist leer – `input_ids` konnte nicht richtig verarbeitet werden!")

    attention_data = []

    # Iteriere über alle Layer und Attention-Heads
       
    for layer_idx, layer in enumerate(attentions):
        for head_idx, head in enumerate(layer):
            generate_diagram(layer_idx + 1, head_idx + 1, tokens, head[0])

            # Falls das Attention-Matrix leer ist, setzen wir eine Standard-Wert
            if head[0].numpy().size == 0:
                max_attention = np.zeros(len(tokens))
            else:
                max_attention = np.max(head[0].numpy(), axis=1)

            # Finde die Indizes der höchsten Attention-Werte (ohne `[CLS]` und `[SEP]`)
            num_tokens = min(3, len(tokens))
            top_indices = np.argsort(max_attention)[-num_tokens:]

            # Sicherstellen, dass nur gültige Tokens berücksichtigt werden
            top_indices = [i for i in top_indices if i < len(tokens) and tokens[i] not in ["[CLS]", "[SEP]"]]

            # Extrahiere die Top-Tokens
            top_tokens = [tokens[i] for i in top_indices]

            # Falls keine validen Tokens gefunden wurden, setze eine Default-Nachricht
            if not top_tokens:
                print(f"⚠️ Warnung: Keine validen Tokens gefunden für Layer {layer_idx+1}, Head {head_idx+1}.")
                top_tokens = ["(keine Tokens)"]

            print(f"✅ Top Tokens für Layer {layer_idx+1}, Head {head_idx+1}: {top_tokens}")

            attention_data.append([layer_idx + 1, head_idx + 1, ", ".join(top_tokens)])
            
    # Speichere die Attention-Daten
    save_attention_data(attention_data)

  

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Erstellt ein Diagramm, das die Attention-Werte zwischen Tokens visualisiert.
    Speichert die Datei als PNG mit Layer- und Head-Nummer im Dateinamen.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Zeichne die Token-Labels
    for i, token in enumerate(tokens):
        try:
            _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        except AttributeError:
            width, _ = draw.textsize(token, font=FONT)

        if width > PIXELS_PER_WORD - 10:
            token = token[:10] + "…"

        draw.text((PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE), token, fill="white", font=FONT)

    # **Fix: Prüfe und passe `attention_weights` an, wenn es zu klein ist**
    min_dim = min(attention_weights.shape[0], len(tokens))
    if attention_weights.shape[0] != len(tokens):
        print(f"⚠️ Achtung: `attention_weights` hat eine unerwartete Form {attention_weights.shape}, erwartet: ({len(tokens)}, {len(tokens)})")
        attention_weights = attention_weights[:min_dim, :min_dim]
        tokens = tokens[:min_dim]
    # Falls die Attention-Matrix kleiner ist, begrenze sie auf die Größe der Tokens
    min_dim = min(attention_weights.shape[0], len(tokens))
    if attention_weights.shape[0] != len(tokens):
        print(f"⚠️ Achtung: `attention_weights` hat eine unerwartete Form {attention_weights.shape}, erwartet: ({len(tokens)}, {len(tokens)})")
        attention_weights = attention_weights[:min_dim, :min_dim]  # Kürze Attention-Matrix
        tokens = tokens[:min_dim]  # Kürze Tokens, falls nötig

    # Zeichne das Gitter mit Attention-Werten
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i, j])  # Hier direkt [i, j] verwenden!
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Speichere das Bild
    filename = f"Attention_Layer{layer_number}_Head{head_number}.png"
    img.save(filename)


if __name__ == "__main__":
    main()



