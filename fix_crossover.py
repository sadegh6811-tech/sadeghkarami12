with open('genetic_engine_boosted.py', 'r') as f:
    content = f.read()

# جایگزینی تابع crossover با یک نسخه امن
old_crossover = '''def crossover(parent1, parent2):
    """ترکیب دو والد (Crossover)."""
    if isinstance(parent1, str) or isinstance(parent2, str):
        return random.choice([parent1, parent2])
    
    op1 = parent1[0]
    op2 = parent2[0]
    
    # با احتمال ۵۰٪، عملگر یکی از والدین را انتخاب کن
    if random.random() < 0.5:
        op = op1
    else:
        op = op2
    
    if op == 'abs':
        child = random.choice([parent1[1], parent2[1]]) if random.random() < 0.5 else parent1[1]
        return ('abs', child)
    elif op == 'max':
        return ('max', 
                random.choice([parent1[1], parent2[1]]),
                random.choice([parent1[2], parent2[2]]))'''

new_crossover = '''def crossover(parent1, parent2):
    """ترکیب دو والد (Crossover) - نسخه امن بدون خطا."""
    # اگر یکی از والدین رشته (متغیر) است
    if isinstance(parent1, str) and isinstance(parent2, str):
        return random.choice([parent1, parent2])
    if isinstance(parent1, str):
        return parent2
    if isinstance(parent2, str):
        return parent1
    
    # هر دو tuple هستند
    op1, op2 = parent1[0], parent2[0]
    
    # اگر یکی abs است و دیگری max، یکی را تصادفی انتخاب کن
    if op1 != op2:
        return random.choice([parent1, parent2])
    
    if op1 == 'abs':
        # هر دو abs دارند، فرزند از ترکیب فرزندانشان
        return ('abs', random.choice([parent1[1], parent2[1]]))
    elif op1 == 'max':
        # هر دو max دارند (دارای ۳ عضو هستند)
        return ('max', 
                random.choice([parent1[1], parent2[1]]),
                random.choice([parent1[2], parent2[2]]))
    
    return parent1'''

if old_crossover in content:
    content = content.replace(old_crossover, new_crossover)
    with open('genetic_engine_boosted.py', 'w') as f:
        f.write(content)
    print("[+] تابع crossover با موفقیت اصلاح شد.")
else:
    print("[!] الگوی crossover پیدا نشد.")
