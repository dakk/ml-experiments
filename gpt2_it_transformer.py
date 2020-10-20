from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead


tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
			
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")
			
		
