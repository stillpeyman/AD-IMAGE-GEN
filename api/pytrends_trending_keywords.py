from pytrends.request import TrendReq
import json
import os
import pandas as pd
import time
import random
import requests


def load_kw_categorization(file_path):
    """
    Load and parse the keywords categorization JSON file.
    
    Args:
        file_path(str): Path to the keyword categorization JSON file
    
    Returns:
        dict: Parsed JSON data with categorized keywords
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def clean_keywords(keywords):
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        kw = kw.lower()
        kw = kw.replace("´", "'")
        kw = kw.replace("—", "-")
        kw = kw.replace("  ", " ")
        if kw:
            cleaned.append(kw)
    return cleaned


def fetch_trends_per_cat(keyword_list, timeframe):
    """
    Fetch Google Trends data for a list of keywords in batches of 5 (Google Trends limit)

    Args:
        keyword_list (list): List of keywords to analyze
        timeframe (str): Time period for trends analysis
    
    Returns:
        list: Combined list of dictionaries with trend data records
    """
    all_records = []

    keyword_list = clean_keywords(keyword_list)

    for i in range(0, len(keyword_list), 5):
        batch = keyword_list[i:i+5]
        print(f"\nFetching batch: {batch}\n")

        # Retry logic (max 3 attempts)
        attempts = 0
        while attempts < 3:
            try:
                pytrends = TrendReq()
                pytrends.build_payload(batch, timeframe=timeframe)

                # Get trend data, Pytrend using interest_over_time returns Pandas DataFrame
                df = pytrends.interest_over_time()

                # Remove the "isPartial" column if it exists
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                
                # Pytrend uses "date" as index, reset_index() turns "date" into a column
                df = df.reset_index()

                # Use "assign type" to convert timestamp to str for JSON serialization
                # df["date"] accesses a Panda Series (e.g. column of data) in DataFrame
                df["date"] = df["date"].astype(str)

                # Convert Pandas DataFrame (df) into list of dicts (records)
                # How? <orient="records"> => each row is a dict: {"column_name": value}
                # Best format for JSON, and for iterating row-by-row
                batch_records = df.to_dict(orient="records")
                if batch_records:
                    # print(batch_records)
                    all_records.extend(batch_records)
                    break                    
            
            except Exception as e:
                print(
                    f"Error fetching batch {batch} (Attempt {attempts + 1}/3): {e}"
                      )
                attempts += 1
                # Wait between requests
                time.sleep(5 + attempts * 2)
        
        # Wait before next batch regardless of retry outcome
        time.sleep(random.uniform(3, 7))

    return all_records


def analyze_keyword_trend(records, keyword, treshold, trend_weight, volume_weight):
    """
    Analyze if a keyword is trending up, down, or stable, and compute a priority score.
    
    Args:
        records (list): Trend records (from fetch_trends_per_cat)
        keyword (str): The keyword to analyze
        threshold (float): Sensitivity for trend detection (e.g. 0.15 = 15% change)
        trend_weight (float): Weight for trend direction in priority score
        volume_weight (float): Weight for volume in priority score

    Returns:
        dict: Analysis with trend, volume, and priority score
    """
    # Extract values for this keyword from records
    values = [record[keyword] for record in records if keyword in record]

    # Need minimum data points
    if len(values) < 6:
        return {"keyword": keyword, "trend": "insufficient_data"}
    
    # Split data into early and recent periods (first/last third)
    total_points = len(values)
    # max(1, ...) takes larger of two values, ensuring result never < 1
    split_point = max(1, total_points // 3)

    early_period = values[:split_point]
    recent_period = values[-split_point:]

    # Calculate averages
    early_avg = sum(early_period) / len(early_period)
    recent_avg = sum(recent_period) / len(recent_period)

    # Avoid division by zero
    # Maybe kw has no search volume 
    # Or new kw with no historical data
    if early_avg == 0:
        return{"keyword": keyword, "trend": "no_early_data"}
    
    # Calculate change ratio
    change_ratio = recent_avg / early_avg
    # Calculate average search interest score
    avg_volume = sum(values) / len(values)

    # Determine trend with chosen threshold
    if change_ratio > 1 + treshold:
        trend = "increasing"
        trend_score = 1

    elif change_ratio < 1 - treshold:
        trend = "decreasing"
        trend_score = 0

    else:
        trend = "stable"
        trend_score = 0.5
    
    # Priority = weighted combo of trend direction and average volume
    # Normalize volume to range 0–1 by dividing by 100 (Google Trends scale goes from 0 to 100)
    volume_score = avg_volume / 100

    # Principle: Weighted Linear Combination (weighted average)
    # Score closer to 1 = strong upward trend and/or high search volume
    priority_score = round(
        trend_weight * trend_score + volume_weight * volume_score, 3
        )
    
    return {
        "keyword": keyword,
        "trend": trend,
        "early_avg": round(early_avg, 2),
        "recent_avg": round(recent_avg, 2),
        "change_ratio": round(change_ratio, 2),
        "avg_volume": round(avg_volume, 2),
        "priority_score": priority_score
    }


def main():
    """
    Main function to execute the trend analysis process.
    """
    # File path to keywords_categorization.json
    kw_file = os.path.join(os.path.dirname(__file__), "..", "data", "keywords_categorization.json")

    data = load_kw_categorization(kw_file)

    # Fetch trends data for each category
    trends_data = {}
    for category, keyword_list in data.items():
        # Skip empty lists
        if not keyword_list:
            print(f"\nSkipping empty category: {category}")
            continue
        
        try:
            records = fetch_trends_per_cat(keyword_list, timeframe="today 3-m")
            # Each category with a list or trend data (records)
            trends_data[category] = records
            print(f"\nSuccessfully fetched data for {category}.")

        except Exception as e:
            print(f"\nError fetching data for {category}: {e}")
            trends_data[category] = []
    
    # Save trends data to JSON
    with open("data/trends_data.json", "w", encoding="utf-8") as handle:
        json.dump(trends_data, handle, indent=2)
    print(f"\nTrends data succesfully saved.")

    print("\n" + "=" * 50)
    print("ANALYZING TRENDS FOR EACH CATEGORY")
    print("=" * 50)

    trending_keywords = {}

    # Analyze trends for each category
    for category, records in trends_data.items():
        # Skip empty categories
        if not records:
            print(f"Skipping empty category: {category}")
            continue

        print(f"\nAanalyzing category: {category}")
        
        # Extract all unique keywords from records (excluding "date")
        # Each record is a row with keyword names as keys
        # Use set to collect all keywords across all rows (batches)
        # Finally use sorted() to turn set into a sorted list
        keywords = sorted(
            {
                k for record in records 
                for k in record.keys() 
                if k != "date"
            }
        )

        increasing_keywords = []
        decreasing_keywords = []
        stable_keywords = []
        missing_keywords = []
        category_analysis = []

        for keyword in keywords:
            analysis = analyze_keyword_trend(records, keyword, treshold=0.15, trend_weight=0.6, volume_weight=0.4)
            category_analysis.append(analysis)

            # Sort keywords by priority descending
            category_analysis.sort(key=lambda item: item.get("priority_score", 0), reverse=True)

            # Print analysis for each keyword
            trend = analysis["trend"]
            if trend == "increasing":
                print(f"""
{keyword}: {trend} 
| ratio: {analysis.get('change_ratio')}x 
| vol: {analysis.get('avg_volume')} 
| priority: {analysis.get('priority_score')}
""")
                increasing_keywords.append(keyword)

            elif trend == "decreasing":
                print(f"""
{keyword}: {trend} 
| ratio: {analysis.get('change_ratio')}x 
| vol: {analysis.get('avg_volume')} 
| priority: {analysis.get('priority_score')}
""")
                decreasing_keywords.append(keyword)

            elif trend == "stable":
                print(f"""
{keyword}: {trend} 
| ratio: {analysis.get('change_ratio')}x 
| vol: {analysis.get('avg_volume')} 
| priority: {analysis.get('priority_score')}
""")
                stable_keywords.append(keyword)

            else:
                print(f"{keyword}: {trend}")
                missing_keywords.append(keyword)
        
        # Create trending_keywords dict, each category is a key
        trending_keywords[category] = {
            "increasing_keywords": increasing_keywords,
            "decreasing_keywords": decreasing_keywords,
            "stable_keywords": stable_keywords,
            "missing_or_skipped": missing_keywords,
            "full_analysis": category_analysis
        }

    # Summary to print in terminal
    print("\n" + "=" * 50)
    print("TRENDING KEYWORDS SUMMARY")
    print("=" * 50)

    for category, data in trending_keywords.items():
        increasing = data["increasing_keywords"]
        decreasing = data["decreasing_keywords"]
        stable = data["stable_keywords"]
        missing = data["missing_or_skipped"]

        print(f"\n{category}:")

        if increasing:
            print(f"increasing: {", ".join(increasing)}")

        if decreasing:
            print(f"decreasing: {", ".join(decreasing)}")
        
        if stable:
            print(f"stable: {", ".join(stable)}")
        
        if missing:
            print(f"missing: {", ".join(missing)}")

        if not (increasing or decreasing or stable or missing):
            print("No keywords found.")
    
    # Save trending_keywords to JSON 
    with open("data/trending_keywords.json", "w", encoding="utf-8") as handle:
        json.dump(trending_keywords, handle, indent=2)
    print("\nFiltered trending keywords saved to data/trending_keywords.json")


if __name__=="__main__":
    main()



    









