import openai
import logging
import re

logger = logging.getLogger(__name__)

def initialize_openai(api_key):
    openai.api_key = api_key
    logger.info("OpenAI API initialized")

def analyze_sentiment(text):
    prompt = (
        "Analyze the sentiment of the following text and provide only a numerical score "
        "between -1 (very negative) and 1 (very positive). Do not include any other text or explanation.\n\n"
        f"Text: {text}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="01-mini",  # Use a valid and accessible model
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_completion_tokens=60,              # Limit tokens to ensure concise output
                     # Set temperature to 0 for deterministic results
        )
        
        # Log the full response for debugging
        logger.debug(f"OpenAI API response: {response}")
        
        # Extract the content safely
        if response and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message'].get('content', '').strip()
            logger.debug(f"Raw sentiment score response: '{content}'")
            
            if content:
                try:
                    sentiment_score = float(content)
                    logger.debug(f"Parsed sentiment score: {sentiment_score}")
                    return sentiment_score
                except ValueError:
                    match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                    if match:
                        sentiment_score = float(match.group())
                        logger.debug(f"Extracted sentiment score via regex: {sentiment_score}")
                        return sentiment_score
                    else:
                        logger.error(f"Unable to parse sentiment score from response: '{content}'")
                        return 0.0  # Default neutral sentiment score
            else:
                logger.error("OpenAI API returned an empty sentiment score.")
                return 0.0
        else:
            logger.error(f"Unexpected OpenAI API response structure: {response}")
            return 0.0
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return 0.0
