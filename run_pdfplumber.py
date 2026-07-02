def extract_text_with_plumber(page_plumber):
    """
    使用 pdfplumber 快速提取纯文本并返回字符串
    """
    extracted_text = page_plumber.extract_text()
    
    if extracted_text:
        print(f"   -> ⚡ Fast extract success! (Characters: {len(extracted_text)})")
        return extracted_text
    else:
        print(f"   -> ⚠️ No text content found by pdfplumber!")
        return ""