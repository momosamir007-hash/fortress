class TeamDictionary:
    @staticmethod
    def get_closest_team(raw_name, teams_list):
        raw_lower = raw_name.lower().strip()
        
        # قاموس الترجمة الشامل لربط أسماء الـ API بأسماء قاعدة البيانات
        aliases = {
            "wolverhampton wanderers": "Wolves",
            "wolverhampton": "Wolves",
            "manchester united": "Man United",
            "manchester city": "Man City",
            "tottenham hotspur": "Tottenham",
            "spurs": "Tottenham",
            "nottingham forest": "Nott'm Forest",
            "sheffield united": "Sheff Utd",
            "west ham united": "West Ham",
            "newcastle united": "Newcastle",
            "brighton & hove albion": "Brighton",
            "brighton and hove albion": "Brighton",
            "leicester city": "Leicester",
            "leeds united": "Leeds",
            "ipswich town": "Ipswich",
            "luton town": "Luton",
            "crystal palace": "Crystal Palace",
            "aston villa": "Aston Villa",
            "bournemouth": "Bournemouth",
            "afc bournemouth": "Bournemouth",
            "qpr": "QPR",
            "queens park rangers": "QPR",
            "brentford": "Brentford",
            "brentford fc": "Brentford",
            "fulham": "Fulham",
            "fulham fc": "Fulham",
            "everton": "Everton",
            "liverpool": "Liverpool",
            "arsenal": "Arsenal",
            "chelsea": "Chelsea"
        }
        
        # 1. البحث في القاموس أولاً
        if raw_lower in aliases:
            mapped_name = aliases[raw_lower]
            for t in teams_list:
                if t.lower() == mapped_name.lower():
                    return t
                    
        # 2. البحث الجزئي (Fallback)
        for t in teams_list:
            if t.lower() in raw_lower or raw_lower in t.lower():
                return t
                
        # 3. الخيار الافتراضي
        return teams_list[0]
