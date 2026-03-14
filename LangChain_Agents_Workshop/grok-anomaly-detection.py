L1:# Zero-Shot Anomaly Detection in Logs using Phi-4 and Deepseek Models
L2:# This script demonstrates zero-shot classification of log lines as 'normal' or 'anomalous'
L3:# using different prompting techniques: Basic Zero-Shot, Chain-of-Thought (CoT), and Tree-of-Thoughts (ToT).
L4:# It uses sample log data with labels for evaluation.
L5:# Models: microsoft/Phi-3-mini-4k-instruct (as Phi-4 proxy; adjust if Phi-4 is available) and deepseek-ai/deepseek-coder-6.7b-instruct.
L6:# Requires: transformers, torch, pandas, numpy. Install via pip if needed.
L7:# Note: Run on GPU for efficiency. Batch size limited by model context.
L8:
L9:import pandas as pd
L10:import numpy as np
L11:import torch
L12:from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
L13:from typing import List, Dict, Tuple
L14:import re
L15:import warnings
L16:warnings.filterwarnings('ignore')
L17:
L18:# Step 1: Create Sample Log Data
L19:# Generate 100 sample logs: 70 normal, 30 anomalous.
L20:# Anomalous logs indicate incidents like errors, crashes, high load, etc.
L21:np.random.seed(42)
L22:num_samples = 100
L23:normal_logs = [
L24:    "User {id} successfully logged in from IP {ip}.",
L25:    "Routine backup completed at {time}.",
L26:    "Database connection established.",
L27:    "File {file} uploaded successfully.",
L28:    "System metrics: CPU 15%, Memory 20%.",
L29:    "Scheduled task {task} executed.",
L30:    "Network traffic normal.",
L31:    "User query processed in 50ms.",
L32:    "Cache refreshed.",
L33:    "Health check passed."
L34:]
L35:anomalous_logs = [
L36:    "ERROR: System crash detected at {time}.",
L37:    "FATAL: Database connection failed.",
L38:    "WARNING: High CPU usage 95% - potential overload.",
L39:    "ALERT: Unauthorized access attempt from {ip}.",
L40:    "CRITICAL: Disk space full, cannot write logs.",
L41:    "ERROR: Service {service} down.",
L42:    "WARNING: Memory leak detected.",
L43:    "FATAL: Network outage.",
L44:    "ALERT: Intrusion detected.",
L45:    "ERROR: Invalid transaction {tx}."
L46:]
L47:
L48:def generate_sample_logs(n_normal: int, n_anomalous: int) -> pd.DataFrame:
L49:    logs = []
L50:    labels = []
L51:    
L52:    # Generate normal logs
L53:    for _ in range(n_normal):
L54:        log = np.random.choice(normal_logs).format(id=np.random.randint(1,1000), ip=f"192.168.1.{np.random.randint(1,255)}",
L55:                                                   time="2025-09-10 12:00:00", file=f"file_{np.random.randint(1,100)}",
L56:                                                   task=f"task_{np.random.randint(1,10)}", service=f"svc_{np.random.randint(1,5)}",
L57:                                                   tx=f"tx_{np.random.randint(1,1000)}")
L58:        logs.append(log)
L59:        labels.append('normal')
L60:    
L61:    # Generate anomalous logs
L62:    for _ in range(n_anomalous):
L63:        log = np.random.choice(anomalous_logs).format(time="2025-09-10 12:00:00", ip=f"192.168.1.{np.random.randint(1,255)}",
L64:                                                      service=f"svc_{np.random.randint(1,5)}", tx=f"tx_{np.random.randint(1,1000)}")
L65:        logs.append(log)
L66:        labels.append('anomalous')
L67:    
L68:    # Shuffle
L69:    combined = list(zip(logs, labels))
L70:    np.random.shuffle(combined)
L71:    logs, labels = zip(*combined)
L72:    
L73:    df = pd.DataFrame({'log': logs, 'label': labels})
L74:    return df
L75:
L76:# Create sample data
L77:df = generate_sample_logs(70, 30)
L78:print("Sample Data Shape:", df.shape)
L79:print(df.head(10))
L80:
L81:# Step 2: Preprocess Logs
L82:def preprocess_log(log: str) -> str:
L83:    """Clean log: remove timestamps, extra whitespace, normalize."""
L84:    # Remove common timestamp patterns
L85:    log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)
L86:    # Remove IPs roughly
L87:    log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '{IP}', log)
L88:    # Strip and normalize
L89:    log = ' '.join(log.strip().split())
L90:    return log
L91:
L92:df['clean_log'] = df['log'].apply(preprocess_log)
L93:print("\nSample Cleaned Logs:")
L94:print(df[['clean_log', 'label']].head(10))
L95:
L96:# Step 3: Define Prompting Techniques
L97:# Technique 1: Basic Zero-Shot
L98:BASIC_PROMPT = """Classify the following log line as 'normal' or 'anomalous'. Respond only with 'normal' or 'anomalous'.
L99:
L100:Log: {log}
L101:
L102:Classification:"""
L103:
L104:# Technique 2: Chain-of-Thought (CoT) - Step by step reasoning
L105:COT_PROMPT = """You are a log analyst. Analyze the log line step by step to classify it as 'normal' or 'anomalous'.
L106:Anomalous logs indicate errors, warnings, failures, alerts, or unusual events related to incidents.
L107:
L108:Step 1: Identify key elements in the log (e.g., keywords like ERROR, WARNING, FATAL, ALERT, crash, failure).
L109:Step 2: Determine if it describes a routine operation or an issue.
L110:Step 3: Classify as 'normal' or 'anomalous' based on that.
L111:
L112:Log: {log}
L113:
L114:Step 1:
L115:Step 2:
L116:Step 3: Classification:"""
L117:
L118:# Technique 3: Tree-of-Thoughts (ToT) - Simulate multiple experts brainstorming
L119:# This prompts the model to think like multiple experts, brainstorm paths, and select best.
L120:TOT_PROMPT = """You are a team of log analysis experts: Expert A (focus on errors/warnings), Expert B (focus on system metrics/performance), Expert C (focus on security/access).
L121:Use Tree of Thoughts: Brainstorm 3 possible interpretations of the log, then select the best path to classify as 'normal' or 'anomalous'.
L122:Anomalous: indicates potential incident like errors, overloads, intrusions.
L123:Normal: routine operations.
L124:
L125:Log: {log}
L126:
L127:Brainstorm:
L128:- Path 1 (Expert A): {reason_a} -> Classification: ?
L129:- Path 2 (Expert B): {reason_b} -> Classification: ?
L130:- Path 3 (Expert C): {reason_c} -> Classification: ?
L131:
L132:Select best path and final Classification: 'normal' or 'anomalous'."""
L133:
L134:# Note: For ToT, the prompt is structured but model generates the reasons. In practice, model fills in.
L135:
L136:# Technique 4: ReACT-style prompting with pseudo-tools
L137:REACT_PROMPT = """You are a senior log analysis agent using the ReACT framework (Reasoning + Acting) to decide if a log line is 'normal' or 'anomalous'.
L138:You have the following pseudo-tools available:
L139:- CHECK_KEYWORDS: inspect the log for words like ERROR, WARNING, FATAL, ALERT, CRITICAL, fail, crash, leak, outage, intrusion, unauthorized.
L140:- CHECK_SEVERITY: judge if the situation described is severe (e.g., crash, outage, security issue) or routine (e.g., successful, completed, established, normal, health check passed).
L141:- CHECK_CONTEXT: determine whether the log describes a routine operation/metric or a potential incident.
L142:
L143:You will think step by step using Thought, then choose an Action (one of the tools), observe the result, and repeat for a few steps.
L144:At the end, you MUST output a line starting with: Final classification: normal  OR  Final classification: anomalous
L145:
L146:Here are examples:
L147:
L148:Example 1 (normal):
L149:Thought: I should see if this is a routine successful operation.
L150:Action: CHECK_KEYWORDS
L151:Observation: The log mentions \"successfully\" and no error-like keywords.
L152:Thought: This looks like a routine successful login, so it is normal.
L153:Final classification: normal
L154:
L155:Example 2 (anomalous):
L156:Thought: I should look for strong error indicators.
L157:Action: CHECK_KEYWORDS
L158:Observation: The log contains \"ERROR\" and \"System crash\" which indicate a failure.
L159:Thought: A crash is a severe incident, so this is anomalous.
L160:Final classification: anomalous
L161:
L162:Now analyze the target log.
L163:
L164:Log: {log}
L165:
L166:Thought:
L167:Action:
L168:Observation:
L169:Thought:
L170:Action:
L171:Observation:
L172:Thought:
L173:Final classification:"""
L174:
L175:# Technique 5: Few-Shot prompting with labeled examples
L176:FEWSHOT_PROMPT = """You are an expert log classifier. Based on the labeled examples, classify the final log as 'normal' or 'anomalous'.
L177:
L178:Here are examples:
L179:
L180:Example 1:
L181:Log: User 123 successfully logged in from IP 10.0.0.5.
L182:Label: normal
L183:
L184:Example 2:
L185:Log: Routine backup completed at 2025-09-10 03:00:00.
L186:Label: normal
L187:
L188:Example 3:
L189:Log: ERROR: System crash detected at 2025-09-10 12:00:00.
L190:Label: anomalous
L191:
L192:Example 4:
L193:Log: ALERT: Unauthorized access attempt from 203.0.113.42.
L194:Label: anomalous
L195:
L196:Example 5:
L197:Log: System metrics: CPU 20%, Memory 35%.
L198:Label: normal
L199:
L200:Now classify the following log. Respond with exactly one word: either 'normal' or 'anomalous'.
L201:
L202:Log: {log}
L203:
L204:Classification:"""
L205:
L206:# Technique 6: Meta prompting (prompt optimizer)
L207:# The model first writes an optimized *classification prompt* tailored to the given log,
L208:# then we run a second generation to get the final label.
L209:META_PROMPT_OPTIMIZER = """You are a prompt engineer optimizing a prompt for zero-shot log anomaly detection.
L210:
L211:Your job: write the BEST possible classification prompt for an LLM to label a single log line as either 'normal' or 'anomalous'.
L212:
L213:Constraints for your output:
L214:- Output ONLY the prompt text (no extra commentary).
L215:- The prompt MUST contain the placeholder {{log}} exactly once, which will be replaced with the log line.
L216:- The prompt MUST instruct the classifier to respond with exactly one word: normal or anomalous.
L217:- Keep it concise.
L218:
L219:Here is the log that the prompt will be used on:
L220:{log}
L221:"""
L222:
L223:# Explanation Prompt (for anomalous logs)
L224:EXPLANATION_PROMPT = """You previously classified this log as anomalous. Provide a 2-line reason why it indicates an anomaly (e.g., error, failure, unusual event).
L225:
L226:Log: {log}
L227:
L228:Reason:"""
L229:
L230:# Step 4: Load Models and Tokenizers
L231:# Note: Use GPU if available. Phi-4 assumed as Phi-3-mini for now; replace with actual Phi-4 model if available.
L232:MODEL_NAMES = {
L233:    'phi': 'microsoft/Phi-3-mini-4k-instruct',  # Proxy for Phi-4; update to 'microsoft/Phi-4-...' if exists
L234:    'deepseek': 'deepseek-ai/deepseek-coder-6.7b-instruct',
L235:    'deepseek7b': 'deepseek-ai/deepseek-coder-7b-instruct'
L236:}
L237:
L238:def load_model(model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
L239:    tokenizer = AutoTokenizer.from_pretrained(model_name)
L240:    if tokenizer.pad_token is None:
L241:        tokenizer.pad_token = tokenizer.eos_token
L242:    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
L243:                                                 device_map='auto' if device == 'cuda' else None)
L244:    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1,
L245:                    max_new_tokens=100, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
L246:    return pipe
L247:
L248:devices = {name: 'cuda' if torch.cuda.is_available() else 'cpu' for name in MODEL_NAMES.keys()}
L249:pipes = {name: load_model(MODEL_NAMES[name], devices[name]) for name in MODEL_NAMES.keys()}
L250:
L251:print("\nModels Loaded:", list(pipes.keys()))
L252:
L253:# Step 5: Prediction Function
L254:def generate_response(pipe, prompt: str, max_tokens: int = 50) -> str:
L255:    """Generate response from model."""
L256:    response = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)[0]['generated_text']
L257:    # Extract after the prompt
L258:    return response[len(prompt):].strip()
L259:
L260:def classify_batch(logs: List[str], technique: str = 'basic', model_name: str = 'phi') -> List[str]:
L261:    """Classify batch of logs using specified technique and model."""
L262:    pipe = pipes[model_name]
L263:    prompt_template = {
L264:        'basic': BASIC_PROMPT,
L265:        'cot': COT_PROMPT,
L266:        'tot': TOT_PROMPT,
L267:        'react': REACT_PROMPT,
L268:        'fewshot': FEWSHOT_PROMPT,
L269:        'meta': None
L270:    }[technique]
L271:    
L272:    classifications = []
L273:    for log in logs:
L274:        if technique == "meta":
L275:            optimizer_prompt = META_PROMPT_OPTIMIZER.format(log=log)
L276:            optimized_prompt = generate_response(pipe, optimizer_prompt, max_tokens=220).strip()
L277:
L278:            # Ensure the optimized prompt can be formatted with {log}.
L279:            # If the model failed to include it, fall back to BASIC_PROMPT.
L280:            if "{log}" not in optimized_prompt:
L281:                prompt = BASIC_PROMPT.format(log=log)
L282:            else:
L283:                prompt = optimized_prompt.format(log=log)
L284:
L285:            response = generate_response(pipe, prompt)
L286:        else:
L287:            prompt = prompt_template.format(log=log)
L288:            response = generate_response(pipe, prompt)
L289:
L290:        # Parse classification: prioritize an explicit "Final classification" line if present (for ReACT),
L291:        # otherwise fall back to searching the whole response.
L292:        classification = None
L293:        lower_response = response.lower()
L294:
L295:        # Look for final classification pattern
L296:        if "final classification" in lower_response:
L297:            # Take the last occurrence to respect the final decision
L298:            last_idx = lower_response.rfind("final classification")
L299:            tail = lower_response[last_idx:]
L300:            if "anomalous" in tail:
L301:                classification = "anomalous"
L302:            elif "normal" in tail:
L303:                classification = "normal"
L304:
L305:        # Generic fallback based on presence anywhere in the response
L306:        if classification is None:
L307:            if 'anomalous' in lower_response:
L308:                classification = 'anomalous'
L309:            elif 'normal' in lower_response:
L310:                classification = 'normal'
L311:            else:
L312:                classification = 'normal'
L313:
L314:        classifications.append(classification)
L315:    
L316:    return classifications
L317:
L318:def get_explanations(anomalous_logs: List[str], model_name: str = 'phi') -> List[str]:
L319:    """Get 2-liner explanations for anomalous logs."""
L320:    pipe = pipes[model_name]
L321:    explanations = []
L322:    for log in anomalous_logs:
L323:        prompt = EXPLANATION_PROMPT.format(log=log)
L324:        response = generate_response(pipe, prompt, max_tokens=100)
L325:        explanations.append(response[:200])  # Truncate to ~2 lines
L326:    return explanations
L327:
L328:# Step 6: Run Experiments
L329:techniques = ['basic', 'fewshot', 'cot', 'tot', 'react', 'meta']
L330:results = {}
L331:
L332:for model_name in MODEL_NAMES.keys():
L333:    results[model_name] = {}
L334:    for technique in techniques:
L335:        print(f"\nRunning {technique.upper()} on {model_name}...")
L336:        
L337:        # Classify all logs
L338:        predictions = classify_batch(df['clean_log'].tolist(), technique, model_name)
L339:        df[f'pred_{model_name}_{technique}'] = predictions
L340:        
L341:        # Find predicted anomalous
L342:        anomalous_mask = [p == 'anomalous' for p in predictions]
L343:        pred_anomalous_logs = df.loc[anomalous_mask, 'clean_log'].tolist()
L344:        
L345:        # Get explanations if any
L346:        if pred_anomalous_logs:
L347:            explanations = get_explanations(pred_anomalous_logs, model_name)
L348:            print(f"Predicted {len(pred_anomalous_logs)} anomalous logs. Sample explanation: {explanations[0] if explanations else 'N/A'}")
L349:        else:
L350:            explanations = []
L351:        
L352:        # Evaluate accuracy (since we have labels)
L353:        true_anomalous = df['label'] == 'anomalous'
L354:        pred_anomalous = np.array(predictions) == 'anomalous'
L355:        accuracy = np.mean(true_anomalous == pred_anomalous)
L356:        precision = np.sum(true_anomalous & pred_anomalous) / np.sum(pred_anomalous) if np.sum(pred_anomalous) > 0 else 0
L357:        recall = np.sum(true_anomalous & pred_anomalous) / np.sum(true_anomalous) if np.sum(true_anomalous) > 0 else 0
L358:        
L359:        results[model_name][technique] = {
L360:            'accuracy': accuracy,
L361:            'precision': precision,
L362:            'recall': recall,
L363:            'num_anomalous_pred': len(pred_anomalous_logs),
L364:            'explanations': explanations
L365:        }
L366:        
L367:        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
L368:
L369:# Step 7: Compare Techniques
L370:print("\n=== RESULTS COMPARISON ===")
L371:rows = []
L372:for model_name, by_technique in results.items():
L373:    for technique, metrics in by_technique.items():
L374:        rows.append({
L375:            "model": model_name,
L376:            "technique": technique,
L377:            "accuracy": metrics.get("accuracy"),
L378:            "precision": metrics.get("precision"),
L379:            "recall": metrics.get("recall"),
L380:            "num_anomalous_pred": metrics.get("num_anomalous_pred"),
L381:        })
L382:
L383:comparison_df = pd.DataFrame(rows).sort_values(["model", "technique"]).reset_index(drop=True)
L384:print(comparison_df.round(3))
L385:
L386:print("\n=== PIVOT (accuracy) ===")
L387:print(comparison_df.pivot(index="model", columns="technique", values="accuracy").round(3))
L388:
L389:# Save results to CSV for notebook viewing
L390:df.to_csv('log_predictions.csv', index=False)
L391:print("\nPredictions saved to 'log_predictions.csv'")
L392:print("\nSample Anomalous Predictions (using CoT on Phi):")
L393:phi_cot_anom_mask = df['pred_phi_cot'] == 'anomalous'
L394:if phi_cot_anom_mask.sum() > 0:
L395:    print(df.loc[phi_cot_anom_mask, ['clean_log', 'label']].head())
L396:
L397:# Note: In a Jupyter notebook, you can add %matplotlib inline and plot results.
L398:# For ToT, the prompt simulates experts; model generates paths internally.
L399:# Adjust batching for large data: use torch.utils.data.DataLoader or loop in chunks.
L400:# For real files: df = pd.read_csv('incident_logs.csv')
