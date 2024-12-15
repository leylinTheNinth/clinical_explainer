import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import gc


def save_lime_explanation(exp, case_info: Dict, save_dir: str = 'explanations'):
    """
    Save LIME explanation with case information
    
    Args:
        exp: LIME explanation object
        case_info: Dictionary containing case details (id, question, correct_option, etc.)
        save_dir: Directory to save explanations
    """
    try:
        # Create timestamp for unique identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = case_info.get('id', 'unknown')
        
        # Create directory structure
        case_dir = os.path.join(save_dir, f"case_{case_id}_{timestamp}")
        os.makedirs(case_dir, exist_ok=True)
        
        # Save LIME explanation
        exp_path = os.path.join(case_dir, 'lime_exp.pkl')
        with open(exp_path, 'wb') as f:
            pickle.dump(exp, f)
            
        # Save HTML visualization
        html_path = os.path.join(case_dir, 'explanation.html')
        exp.save_to_file(html_path)
        
        # Save case information
        info_path = os.path.join(case_dir, 'case_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(case_info, f)
            
        return case_dir
        
    except Exception as e:
        print(f"Error saving explanation for case {case_id}: {str(e)}")
        return None

def load_lime_explanation(case_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load LIME explanation and case information
    
    Args:
        case_dir: Directory containing saved explanation
        
    Returns:
        Dictionary containing loaded explanation and case info
    """
    try:
        # Load LIME explanation
        exp_path = os.path.join(case_dir, 'lime_exp.pkl')
        with open(exp_path, 'rb') as f:
            exp = pickle.load(f)
            
        # Load case information
        info_path = os.path.join(case_dir, 'case_info.pkl')
        with open(info_path, 'rb') as f:
            case_info = pickle.load(f)
            
        return {
            'explanation': exp,
            'case_info': case_info,
            'html_path': os.path.join(case_dir, 'explanation.html')
        }
        
    except Exception as e:
        print(f"Error loading explanation from {case_dir}: {str(e)}")
        return None

