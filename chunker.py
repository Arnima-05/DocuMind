from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

print("Total characters:", len(full_text))

chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)