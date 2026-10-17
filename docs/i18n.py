"""Interface strings and the Persian syllabus text.

build.py renders the site once per language in LANGS_UI. English is the source
language; a Persian page is used when docs/pages/fa/<slug>.html exists, and the
English page is shown (with a notice) when it does not.

To translate a new syllabus entry, add it to FA_CHAPTERS below (same chapter
numbers and section ids as syllabus.py).
"""

LANGS_UI = {
    "en": dict(code="en", dir="ltr", name="English", short="EN", out=""),
    "fa": dict(code="fa", dir="rtl", name="فارسی", short="فا", out="fa/"),
}

UI = {
    "en": dict(
        brand="LLM from scratch",
        site_title="Build an LLM from scratch",
        chapter="Chapter", appendices="Appendices",
        s_ready="ready", s_progress="in progress", s_planned="planned",
        none_yet="none yet",
        h_start="Start", h_reference="Reference", h_source="Source",
        this_repo="This repo", book_repo="Raschka's repo",
        th_book="Book", th_section="Section", th_status="Status", th_topic="Topic", th_written="Written",
        open="open", sections_word="sections",
        companion="Companion code in Raschka's repo",
        code_label="Code:",
        planned_note='Not written yet. The outline below follows the book. To add it, follow <a href="contributing.html">Contributing</a>; the roadmap and navigation update by themselves.',
        sections_h="Sections", extras_h="Extras",
        no_sections='Section list not mapped yet. See the book for the outline.',
        not_impl="Not implemented in this language yet.",
        book_section="Book section", code_in="code in", extra="extra",
        edit="Edit this page on GitHub",
        prev="Previous", next="Next",
        skip="Skip to content", menu="Menu", theme="Toggle theme",
        filter="Filter pages", filter_ph="Filter pages…", no_match="No page matches.",
        toc="On this page", copy="Copy", copied="Copied", lang_switch="Switch language",
        fallback='This page has not been translated into Persian yet. The English original is shown below.',
        untranslated="EN",
        roadmap_title="Roadmap",
        roadmap_body="""
<p class="lede">Where the project is, and what is next. This page is generated from <code>docs/syllabus.py</code> and the page files, so it is always current.</p>
<h2>The book, chapter by chapter</h2>
{progress}
<h2>Code by language</h2>
<p>A tick means that language implements the chapter's code, tested to give the same results as the others.</p>
{matrix}
<h2>How the project grows</h2>
<ol>
<li><b>Cover the book.</b> One chapter at a time, in the book's order. Concepts first, then code in every language.</li>
<li><b>Extras.</b> Deeper dives that the book only hints at: worked examples, comparisons, bonus topics. They live next to their chapter.</li>
<li><b>More languages.</b> Adding a language means filling its folder with the same modules and one column in the tables.</li>
</ol>
<p>Want to help or extend it? <a href="contributing.html">Contributing</a> has the exact steps.</p>
""",
    ),
    "fa": dict(
        brand="ساخت LLM از صفر",
        site_title="ساخت یک LLM از صفر",
        chapter="فصل", appendices="پیوست‌ها",
        s_ready="آماده", s_progress="در حال نگارش", s_planned="برنامه‌ریزی‌شده",
        none_yet="هنوز نیست",
        h_start="شروع", h_reference="مرجع", h_source="منبع",
        this_repo="این مخزن", book_repo="مخزن راشکا",
        th_book="کتاب", th_section="بخش", th_status="وضعیت", th_topic="موضوع", th_written="نوشته‌شده",
        open="باز کردن", sections_word="بخش",
        companion="کد همراه کتاب در مخزن راشکا",
        code_label="کد:",
        planned_note='این بخش هنوز نوشته نشده است. فهرست زیر از روی کتاب است. برای افزودن آن، راهنمای <a href="contributing.html">مشارکت</a> را دنبال کنید؛ نقشهٔ راه و ناوبری خودکار به‌روز می‌شوند.',
        sections_h="بخش‌ها", extras_h="مطالب تکمیلی",
        no_sections='فهرست بخش‌ها هنوز تعیین نشده است. فهرست کامل را در کتاب ببینید.',
        not_impl="این زبان هنوز پیاده‌سازی نشده است.",
        book_section="بخش کتاب", code_in="کد به زبان", extra="تکمیلی",
        edit="ویرایش این صفحه در GitHub",
        prev="قبلی", next="بعدی",
        skip="پرش به محتوا", menu="منو", theme="تغییر پوسته",
        filter="جستجو در صفحه‌ها", filter_ph="جستجو در صفحه‌ها…", no_match="صفحه‌ای پیدا نشد.",
        toc="در این صفحه", copy="کپی", copied="کپی شد", lang_switch="تغییر زبان",
        fallback='این صفحه هنوز به فارسی ترجمه نشده است. متن انگلیسی اصلی در ادامه آمده است.',
        untranslated="EN",
        roadmap_title="نقشهٔ راه",
        roadmap_body="""
<p class="lede">پروژه کجاست و گام بعدی چیست. این صفحه از روی <code>docs/syllabus.py</code> و فایل‌های صفحه‌ها ساخته می‌شود، پس همیشه به‌روز است.</p>
<h2>کتاب، فصل به فصل</h2>
{progress}
<h2>کد به تفکیک زبان</h2>
<p>علامت ✓ یعنی آن زبان کد فصل را پیاده کرده و خروجی‌اش با بقیهٔ زبان‌ها یکسان آزمایش شده است.</p>
{matrix}
<h2>پروژه چگونه رشد می‌کند</h2>
<ol>
<li><b>پوشش کامل کتاب.</b> هر بار یک فصل، به ترتیب کتاب. نخست مفهوم، سپس کد در همهٔ زبان‌ها.</li>
<li><b>مطالب تکمیلی.</b> بررسی‌های عمیق‌تر از آنچه کتاب فقط به آن اشاره می‌کند: مثال‌های گام‌به‌گام، مقایسه‌ها و موضوعات اضافه. کنار فصل مربوط قرار می‌گیرند.</li>
<li><b>زبان‌های بیشتر.</b> افزودن یک زبان یعنی پر کردن پوشه‌اش با همان ماژول‌ها و یک ستون در جدول‌ها.</li>
</ol>
<p>می‌خواهید کمک کنید یا آن را گسترش دهید؟ راهنمای <a href="contributing.html">مشارکت</a> گام‌های دقیق را دارد.</p>
""",
    ),
}

# Persian titles, summaries and section titles, keyed by chapter number.
FA_CHAPTERS = {
    1: dict(title="شناخت مدل‌های زبانی بزرگ",
            summary="LLM چیست، چگونه آموزش می‌بیند و یک GPT چه شکلی دارد. بدون کد.",
            sections={"1.1": "LLM چیست؟", "1.2": "کاربردهای LLM", "1.3": "مراحل ساخت و استفاده از LLM",
                      "1.4": "استفاده از LLM برای وظایف گوناگون", "1.5": "بهره‌گیری از مجموعه‌داده‌های بزرگ",
                      "1.6": "نگاهی دقیق‌تر به معماری GPT", "1.7": "ساختن یک مدل زبانی بزرگ"}),
    2: dict(title="کار با داده‌های متنی",
            summary="تبدیل متن خام به ورودی مدل: توکن‌ها، شناسه‌ها، رمزگذاری جفت‌بایت، جفت‌های آموزشی و امبدینگ‌ها.",
            sections={"2.1": "درک امبدینگ کلمه", "2.2": "توکن‌سازی متن", "2.3": "تبدیل توکن‌ها به شناسه",
                      "2.4": "افزودن توکن‌های ویژهٔ زمینه", "2.5": "رمزگذاری جفت‌بایت",
                      "2.6": "نمونه‌برداری از داده با پنجرهٔ لغزان", "2.7": "ساخت امبدینگ توکن",
                      "2.8": "کدگذاری جایگاه کلمات"}),
    3: dict(title="کدنویسی سازوکارهای توجه",
            summary="چرا توجه (attention) لازم شد؛ سپس خودتوجهی، ماسک علّی، دراپ‌اوت و توجه چندسَری.",
            sections={"3.1": "مشکل مدل‌سازی دنباله‌های بلند",
                      "3.2": "گرفتن وابستگی‌های داده با سازوکار توجه",
                      "3.3": "توجه به بخش‌های مختلف ورودی با خودتوجهی",
                      "3.4": "پیاده‌سازی خودتوجهی با وزن‌های آموزش‌پذیر",
                      "3.5": "پنهان کردن کلمات آینده با توجه علّی",
                      "3.6": "گسترش توجه تک‌سَر به توجه چندسَری"}),
    4: dict(title="پیاده‌سازی مدل GPT از صفر",
            summary="نرمال‌سازی لایه، شبکهٔ پیش‌خور، اتصال میان‌بر، بلوک ترنسفورمر، مدل کامل GPT و تولید متن.",
            sections={"4.1": "کدنویسی معماری LLM", "4.2": "نرمال‌سازی فعال‌سازی‌ها با نرمال‌سازی لایه",
                      "4.3": "پیاده‌سازی شبکهٔ پیش‌خور با فعال‌ساز GELU", "4.4": "افزودن اتصال‌های میان‌بر",
                      "4.5": "اتصال توجه و لایه‌های خطی در بلوک ترنسفورمر", "4.6": "کدنویسی مدل GPT",
                      "4.7": "تولید متن"}),
    5: dict(title="پیش‌آموزش روی داده‌های بدون برچسب",
            summary="تابع زیان، حلقهٔ آموزش، راهبردهای رمزگشایی (دما، top-k)، ذخیرهٔ وزن‌ها و بارگذاری وزن‌های از پیش آموزش‌دیدهٔ OpenAI.",
            sections={"5.1": "ارزیابی مدل‌های مولد متن", "5.2": "آموزش یک LLM",
                      "5.3": "راهبردهای رمزگشایی برای کنترل تصادفی بودن",
                      "5.4": "بارگذاری و ذخیرهٔ وزن‌های مدل در PyTorch",
                      "5.5": "بارگذاری وزن‌های از پیش آموزش‌دیدهٔ OpenAI"}),
    6: dict(title="فاین‌تیون برای دسته‌بندی متن",
            summary="تبدیل GPT از پیش آموزش‌دیده به دسته‌بند هرزنامه: آماده‌سازی داده برچسب‌دار، افزودن سرِ دسته‌بندی، فاین‌تیون و ارزیابی.",
            sections={}),
    7: dict(title="فاین‌تیون برای پیروی از دستورها",
            summary="مجموعه‌داده‌های دستوری، دسته‌بندی با پدینگ، فاین‌تیون و ارزیابی پاسخ‌ها.",
            sections={}),
    "A": dict(title="مطالب همراه",
              summary="الف: آشنایی با PyTorch. ب: منابع. ج: پاسخ تمرین‌ها. د: افزودنی‌های حلقهٔ آموزش (گرم‌کردن، کاهش کسینوسی، برش گرادیان). ه: فاین‌تیون با LoRA.",
              sections={"A": "آشنایی با PyTorch", "B": "منابع و مطالعهٔ بیشتر", "C": "پاسخ تمرین‌ها",
                        "D": "افزودن امکانات به حلقهٔ آموزش",
                        "E": "فاین‌تیون کم‌پارامتر با LoRA"}),
}
