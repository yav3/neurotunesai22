from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


LANGUAGE_CONFIGS: dict[str, dict] = {
    "en": {
        "name": "English",
        "stopwords": frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "shall", "may", "might", "can", "must", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "but", "and", "or", "if", "while", "about", "up",
            "it", "its", "he", "she", "they", "them", "his", "her", "their",
            "we", "us", "our", "you", "your", "i", "me", "my", "this", "that",
            "these", "those", "what", "which", "who", "whom", "am", "also",
            "any", "much", "many", "still", "even", "well", "back", "now",
            "like", "get", "got", "go", "going", "come", "came", "make", "take",
            "know", "think", "see", "look", "want", "give", "use", "find",
            "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
        }),
        "script": "latin",
    },
    "es": {
        "name": "Spanish",
        "stopwords": frozenset({
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
            "en", "y", "que", "es", "por", "con", "para", "como", "pero", "su",
            "al", "se", "lo", "no", "le", "les", "ya", "me", "mi", "te", "tu",
            "nos", "si", "sin", "sobre", "este", "esta", "ese", "esa", "aquel",
            "aquella", "muy", "fue", "ser", "ha", "hay", "son", "era", "más",
            "mas", "todo", "esta", "entre", "cuando", "donde", "puede", "todos",
            "bien", "tiene", "porque", "cada", "poco", "mucho", "desde", "otro",
            "otra", "otros", "otras", "nos", "tan", "yo", "él", "ella", "ellos",
            "nosotros", "ustedes", "sus", "quien", "cual", "hasta", "también",
            "hacia", "así", "después", "entonces", "o", "ni", "algo", "nada",
        }),
        "script": "latin",
        "diminutive_patterns": [r"ito$", r"ita$", r"illo$", r"illa$", r"ico$", r"ica$"],
    },
    "fr": {
        "name": "French",
        "stopwords": frozenset({
            "le", "la", "les", "un", "une", "des", "de", "du", "et", "est",
            "en", "que", "qui", "dans", "ce", "il", "ne", "sur", "se", "pas",
            "plus", "par", "je", "au", "avec", "mais", "on", "son", "sa", "ses",
            "ou", "lui", "nous", "vous", "elle", "leur", "mon", "ton", "notre",
            "votre", "tout", "être", "avoir", "faire", "comme", "pour", "si",
            "bien", "même", "aussi", "peut", "entre", "sans", "après", "avant",
            "sous", "très", "cette", "ces", "mes", "tes", "nos", "vos", "aux",
            "ici", "donc", "car", "ni", "encore", "toujours", "jamais", "rien",
            "peu", "trop", "beaucoup", "alors", "quand", "où", "comment",
            "pourquoi", "quel", "quelle", "y", "à", "ma", "ta",
        }),
        "script": "latin",
    },
    "pt": {
        "name": "Portuguese",
        "stopwords": frozenset({
            "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da",
            "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com",
            "sem", "sob", "sobre", "e", "ou", "mas", "que", "se", "não", "mais",
            "muito", "já", "também", "só", "como", "eu", "tu", "ele", "ela",
            "nós", "eles", "elas", "você", "vocês", "meu", "minha", "seu", "sua",
            "nosso", "nossa", "este", "esta", "esse", "essa", "aquele", "aquela",
            "isto", "isso", "aquilo", "ser", "estar", "ter", "haver", "fazer",
            "ir", "vir", "poder", "dever", "querer", "saber", "ver", "dar",
            "ao", "aos", "à", "às", "até", "entre", "depois", "antes", "ainda",
        }),
        "script": "latin",
        "brazilian_markers": ["você", "vocês", "ônibus", "trem"],
        "european_markers": ["tu", "vós", "autocarro", "comboio"],
    },
    "de": {
        "name": "German",
        "stopwords": frozenset({
            "der", "die", "das", "ein", "eine", "und", "ist", "in", "zu", "den",
            "von", "mit", "auf", "für", "an", "es", "als", "auch", "aus", "bei",
            "nach", "um", "über", "so", "wie", "aber", "oder", "wenn", "noch",
            "wird", "hat", "ich", "er", "sie", "wir", "ihr", "du", "man",
            "nicht", "sind", "war", "kann", "dem", "des", "sich", "haben",
            "sein", "werden", "nur", "schon", "sehr", "doch", "kein", "keine",
            "mein", "dein", "sein", "unser", "euer", "dieser", "diese",
            "dieses", "jeder", "jede", "jedes", "alle", "viel", "mehr",
        }),
        "script": "latin",
    },
    "it": {
        "name": "Italian",
        "stopwords": frozenset({
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "del",
            "della", "dei", "degli", "delle", "in", "a", "da", "con", "su", "per",
            "tra", "fra", "e", "o", "ma", "che", "non", "si", "è", "sono",
            "ha", "ho", "come", "più", "molto", "anche", "già", "ancora",
            "io", "tu", "lui", "lei", "noi", "voi", "loro", "mi", "ti", "ci",
            "vi", "mio", "tuo", "suo", "nostro", "vostro", "questo", "quello",
            "ogni", "tutto", "tutti", "essere", "avere", "fare", "dire",
        }),
        "script": "latin",
    },
    "ja": {
        "name": "Japanese",
        "stopwords": frozenset({
            "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ",
            "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として",
            "い", "や", "れる", "など", "なっ", "ない", "この", "ため", "その",
            "あっ", "よう", "また", "もの", "という", "あり", "まで", "られ",
            "なる", "へ", "か", "だ", "これ", "によって", "により", "おり",
            "より", "による", "ず", "なり", "られる", "において", "ば", "なかっ",
            "なく", "しかし", "について", "せ", "だっ", "その後", "できる",
        }),
        "script": "cjk",
    },
    "ko": {
        "name": "Korean",
        "stopwords": frozenset({
            "이", "그", "저", "것", "수", "등", "들", "및", "에", "의", "가",
            "을", "를", "은", "는", "로", "으로", "와", "과", "도", "에서",
            "까지", "부터", "만", "이다", "하다", "있다", "없다", "되다",
            "않다", "그리고", "하지만", "그러나", "또는", "때문에",
        }),
        "script": "hangul",
    },
    "zh": {
        "name": "Chinese",
        "stopwords": frozenset({
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
            "们", "那", "被", "从", "把", "让", "与", "对", "为", "而",
            "但", "又", "或", "如果", "因为", "所以", "可以", "已经",
        }),
        "script": "cjk",
    },
    "ar": {
        "name": "Arabic",
        "stopwords": frozenset({
            "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك",
            "تلك", "التي", "الذي", "هو", "هي", "هم", "هن", "أنا", "نحن",
            "أنت", "أنتم", "كان", "كانت", "يكون", "لا", "ما", "لم", "لن",
            "قد", "أن", "إن", "أو", "و", "ف", "ب", "ل", "ك", "ثم", "حتى",
            "بين", "عند", "كل", "بعض", "غير", "أي", "كما", "لكن", "بل",
        }),
        "script": "arabic",
    },
    "hi": {
        "name": "Hindi",
        "stopwords": frozenset({
            "का", "के", "की", "में", "है", "हैं", "को", "से", "पर", "ने",
            "और", "या", "एक", "यह", "वह", "इस", "उस", "जो", "कि", "भी",
            "तो", "पर", "अपने", "कर", "हो", "था", "थी", "थे", "गया", "गयी",
            "नहीं", "अब", "जब", "तक", "साथ", "लिए", "कुछ", "बहुत", "सब",
        }),
        "script": "devanagari",
    },
    "ru": {
        "name": "Russian",
        "stopwords": frozenset({
            "и", "в", "не", "на", "я", "что", "он", "с", "это", "а", "как",
            "но", "она", "по", "к", "у", "из", "мы", "за", "вы", "так",
            "же", "от", "все", "то", "бы", "ещё", "да", "нет", "его", "её",
            "их", "мой", "твой", "наш", "ваш", "этот", "тот", "быть", "был",
            "была", "были", "будет", "есть", "для", "только", "или", "ни",
            "чтобы", "если", "уже", "ли", "очень", "когда", "где", "кто",
            "себя", "свой", "при", "между", "тоже", "также", "однако",
        }),
        "script": "cyrillic",
    },
}


_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk", 0x3400, 0x4DBF),
    ("cjk", 0x3040, 0x309F),  # hiragana
    ("cjk", 0x30A0, 0x30FF),  # katakana
    ("hangul", 0xAC00, 0xD7AF),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    ("devanagari", 0x0900, 0x097F),
    ("cyrillic", 0x0400, 0x04FF),
    ("cyrillic", 0x0500, 0x052F),
    ("latin", 0x0041, 0x007A),
    ("latin", 0x00C0, 0x024F),
]


class MultilingualProcessor:

    def detect_script(self, text: str) -> str:
        counts: dict[str, int] = {}
        for ch in text:
            if ch.isspace() or unicodedata.category(ch).startswith("P"):
                continue
            cp = ord(ch)
            for script, lo, hi in _SCRIPT_RANGES:
                if lo <= cp <= hi:
                    counts[script] = counts.get(script, 0) + 1
                    break
        if not counts:
            return "unknown"
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def detect_language(self, text: str) -> tuple[str, float]:
        script = self.detect_script(text)

        if script == "hangul":
            return ("ko", 0.95)
        if script == "arabic":
            return ("ar", 0.90)
        if script == "devanagari":
            return ("hi", 0.90)
        if script == "cyrillic":
            return ("ru", 0.85)
        if script == "cjk":
            has_hiragana = any(0x3040 <= ord(c) <= 0x309F for c in text)
            has_katakana = any(0x30A0 <= ord(c) <= 0x30FF for c in text)
            if has_hiragana or has_katakana:
                return ("ja", 0.90)
            return ("zh", 0.80)

        tokens = self._latin_tokenize(text)
        token_set = {t.lower() for t in tokens}
        if not token_set:
            return ("en", 0.1)

        best_lang = "en"
        best_score = 0.0
        for lang_code, cfg in LANGUAGE_CONFIGS.items():
            if cfg.get("script") != "latin":
                continue
            overlap = len(token_set & cfg["stopwords"])
            score = overlap / max(len(token_set), 1)
            if score > best_score:
                best_score = score
                best_lang = lang_code

        confidence = min(best_score * 3.0, 0.99)
        return (best_lang, round(confidence, 2))

    def tokenize(self, text: str, language: str | None = None) -> list[str]:
        if language is None:
            language, _ = self.detect_language(text)

        cfg = LANGUAGE_CONFIGS.get(language, {})
        script = cfg.get("script", self.detect_script(text))

        if script == "cjk":
            return self._cjk_tokenize(text)
        if script == "hangul":
            return self._hangul_tokenize(text)
        return self._latin_tokenize(text)

    def remove_stopwords(self, tokens: list[str], language: str) -> list[str]:
        cfg = LANGUAGE_CONFIGS.get(language, {})
        stops = cfg.get("stopwords", frozenset())
        return [t for t in tokens if t.lower() not in stops]

    def syllable_count(self, word: str, language: str = "en") -> int:
        if not word:
            return 0

        cfg = LANGUAGE_CONFIGS.get(language, {})
        script = cfg.get("script", "latin")

        if script == "cjk":
            return len(word)
        if script == "hangul":
            return sum(1 for c in word if 0xAC00 <= ord(c) <= 0xD7AF)

        return self._latin_syllable_count(word)

    def _latin_tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿÀ-ɏ]+(?:['']\w+)?", text)

    def _cjk_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        buf = ""
        for ch in text:
            cp = ord(ch)
            is_cjk = (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                       or 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF)
            if is_cjk:
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            elif ch.isalnum():
                buf += ch
            else:
                if buf:
                    tokens.append(buf)
                    buf = ""
        if buf:
            tokens.append(buf)
        return tokens

    def _hangul_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        buf = ""
        for ch in text:
            cp = ord(ch)
            is_hangul = (0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF
                         or 0x3130 <= cp <= 0x318F)
            if is_hangul or ch.isalnum():
                buf += ch
            else:
                if buf:
                    tokens.append(buf)
                    buf = ""
        if buf:
            tokens.append(buf)
        return tokens

    @staticmethod
    def _latin_syllable_count(word: str) -> int:
        word = word.lower().strip()
        if len(word) <= 2:
            return 1
        vowels = "aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ"
        count = 0
        prev_vowel = False
        for ch in word:
            if ch in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        if word.endswith("e") and count > 1 and not word.endswith("le"):
            count -= 1
        return max(count, 1)
