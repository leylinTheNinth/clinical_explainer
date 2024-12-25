import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Any
import torch
import gc
import shap


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

def save_shap_explanation(exp: Any, 
                         case_info: Dict,
                         save_dir: str = "explanations") -> str:
    """
    Save SHAP explanation and case information.
    
    Args:
        exp: SHAP explanation object
        case_info: Dictionary containing case information
        save_dir: Directory to save explanations
    
    Returns:
        str: Path where explanation was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_id = case_info['id']
    base_filename = f"shap_explanation_{case_id}_{timestamp}"
    base_path = os.path.join(save_dir, base_filename)
    
    try:
        # Save as pickle (complete object)
        with open(f"{base_path}.pkl", 'wb') as f:
            pickle.dump({
                'shap_values': exp,
                'case_info': case_info
            }, f)
        shap.plots.text(exp)
        return base_path
        
    except Exception as e:
        print(f"Error saving SHAP explanation: {str(e)}")
        raise

def load_shap_explanation(base_path: str) -> Dict:
    """
    Load SHAP explanation from saved files.
    
    Args:
        base_path: Base path of saved explanation files
        
    Returns:
        Dict: Dictionary containing loaded explanation components
    """
    try:
        # Load from pickle
        with open(f"{base_path}.pkl", 'rb') as f:
            data = pickle.load(f)
            
        return {
            'shap_values': data['shap_values'],
            'case_info': data['case_info'],
            'plot_path': f"{base_path}_plot.png"
        }
        
    except Exception as e:
        print(f"Error loading SHAP explanation: {str(e)}")
        raise

def save_explanation(exp: Any, 
                    case_info: Dict, 
                    explainer_type: str,
                    save_dir: str = "explanations") -> str:
    """
    Generic function to save explanations based on type.
    
    Args:
        exp: Explanation object (LIME or SHAP)
        case_info: Dictionary containing case information
        explainer_type: Type of explainer ('lime' or 'shap')
        save_dir: Directory to save explanations
        
    Returns:
        str: Path where explanation was saved
    """
    if explainer_type == 'lime':
        return save_lime_explanation(exp, case_info, save_dir)
    elif explainer_type == 'shap':
        return save_shap_explanation(exp, case_info, save_dir)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    
def save_decoder_outputs(token_shap_exp: Any, 
                        case_info: Dict,
                        prediction: Dict,
                        model_name: str,  # Add model_name parameter
                        save_dir: str = "explanations") -> str:
    """
    Save both TokenSHAP explanation and model prediction for decoder models.
    Args:
        token_shap_exp: TokenSHAP object with analysis
        case_info: Dict containing case details
        prediction: Dict containing model's prediction
        model_name: Name/path of the model used
        save_dir: Directory to save outputs
    
    Returns:
        str: Path where outputs were saved
    """
    case_id = case_info['id']
    model_short_name = model_name.split('/')[-1]  # Get last part of model path
    case_dir = os.path.join(save_dir, f"decoder_case_{case_id}_{model_short_name}")
    os.makedirs(case_dir, exist_ok=True)
    
    try:
        print("Saving Token Shap values and case information...")
        essential_shap_data = {
            'shapley_values': token_shap_exp.shapley_values,
            'tokens': token_shap_exp.baseline_text,
            'model_name': model_name  
        }
        
        exp_path = os.path.join(case_dir, 'token_shap.pkl')
        with open(exp_path, 'wb') as f:
            pickle.dump(essential_shap_data, f)
        
        # Save prediction with model information
        prediction_data = {
            'prediction': prediction,
            'model_name': model_name
        }
        pred_path = os.path.join(case_dir, 'prediction.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump(prediction_data, f)
            
        # Save case info
        info_path = os.path.join(case_dir, 'case_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(case_info, f)

        print(f"Successfully saved SHAP explanation for case {case_id}")
        print(f"Model used: {model_name}")
        return case_dir
        
    except Exception as e:
        print(f"Error saving decoder outputs: {str(e)}")
        raise

def load_decoder_outputs(case_dir: str) -> Dict:
    """
    Load saved decoder outputs (TokenSHAP essentials and prediction)
    
    Args:
        case_dir: Directory containing saved decoder outputs
        
    Returns:
        Dict containing:
            - token_shap: Dict with shapley_values and tokens
            - prediction: Dict with model response
            - case_info: Original case information
            - model_name: Name of the model used
    """
    try:
        # Load TokenSHAP essential data
        exp_path = os.path.join(case_dir, 'token_shap.pkl')
        with open(exp_path, 'rb') as f:
            token_shap_data = pickle.load(f)
            
        # Load prediction
        pred_path = os.path.join(case_dir, 'prediction.pkl')
        with open(pred_path, 'rb') as f:
            prediction_data = pickle.load(f)
            
        # Load case info
        info_path = os.path.join(case_dir, 'case_info.pkl')
        with open(info_path, 'rb') as f:
            case_info = pickle.load(f)
            
        # Get model name from token_shap_data
        model_name = token_shap_data.get('model_name', 'Unknown Model')
            
        return {
            'token_shap': {
                'shapley_values': token_shap_data['shapley_values'],
                'tokens': token_shap_data['tokens']
            },
            'prediction': prediction_data['prediction'],
            'case_info': case_info,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"Error loading decoder outputs from {case_dir}: {str(e)}")
        raise