// The streamed frames of tests/test_stream_golden.cpp, taken from fcc9e41e (the
// streamed routes before the raw trace reordered their last steps; the error
// frames of a throwing turn too), response and request ids and timestamps
// normalised. Regenerate only for a change that means to move them.
const std::map<std::string, std::string>& golden() {
    static const std::map<std::string, std::string> g = {
        {"chat_held_closer_tail", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"plan "},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"</th"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_closed":false},"finish_reason":"length","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":2,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":2}}

data: [DONE]

)GOLDEN"},
        {"chat_length_in_call", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Let me.\n"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"<tool_call>\n<function=run_q"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{},"finish_reason":"length","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":3,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":3}}

data: [DONE]

)GOLDEN"},
        {"chat_length_in_reasoning", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"weigh "},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"it "},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"ét"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_closed":false},"finish_reason":"length","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":3,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":3}}

data: [DONE]

)GOLDEN"},
        {"chat_stop_cut", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Paris"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" "},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":2,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":2}}

data: [DONE]

)GOLDEN"},
        {"chat_stop_cut_tools", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Paris"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" "},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":3,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":3}}

data: [DONE]

)GOLDEN"},
        {"chat_think_call", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"weigh it"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Running it.\n"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"sql\":\"SELECT 1 WHERE a < 2\"}","name":"run_query"},"id":"call_0","index":0,"type":"function"}]},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_closed":true},"finish_reason":"tool_calls","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":7,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":7}}

data: [DONE]

)GOLDEN"},
        {"chat_think_call_notools", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"weigh it"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Running it.\n"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"<tool_call>"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"</tool_call>"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_closed":true},"finish_reason":"stop","index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk","usage":{"completion_tokens":7,"completion_tokens_details":{"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"prompt_tokens":0,"total_tokens":7}}

data: [DONE]

)GOLDEN"},
        {"chat_throws_mid_answer", R"GOLDEN(data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"reasoning_content":"weigh it"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"half an ans"},"finish_reason":null,"index":0}],"created":0,"id":"X","model":"stub","object":"chat.completion.chunk"}

data: {"error":{"code":null,"message":"backend exploded","param":null,"type":"internal_error"}}

data: [DONE]

)GOLDEN"},
        {"responses_held_closer_tail", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"text":"","type":"reasoning_text"}],"id":"X","status":"in_progress","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"","type":"reasoning_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"plan ","item_id":"X","output_index":0,"sequence_number":5,"type":"response.reasoning.delta"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"</th","item_id":"X","output_index":0,"sequence_number":6,"type":"response.reasoning.delta"}

event: response.reasoning.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":7,"text":"plan </th","type":"response.reasoning.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"plan </th","type":"reasoning_text"},"sequence_number":8,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"text":"plan </th","type":"reasoning_text"}],"id":"X","status":"incomplete","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":9,"type":"response.output_item.done"}

event: response.incomplete
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":{"reason":"max_output_tokens"},"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"text":"plan </th","type":"reasoning_text"}],"id":"X","status":"incomplete","summary":[],"type":"reasoning"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"incomplete","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":{"input_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}},"sequence_number":10,"type":"response.incomplete"}

data: [DONE]

)GOLDEN"},
        {"responses_length_in_call", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"annotations":[],"text":"","type":"output_text"}],"id":"X","role":"assistant","status":"in_progress","type":"message"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.output_text.delta
data: {"content_index":0,"delta":"Let me.\n","item_id":"X","output_index":0,"sequence_number":5,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"<tool_call>\n<function=run_q","item_id":"X","output_index":0,"sequence_number":6,"type":"response.output_text.delta"}

event: response.output_text.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":7,"text":"Let me.\n<tool_call>\n<function=run_q","type":"response.output_text.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"annotations":[],"text":"Let me.\n<tool_call>\n<function=run_q","type":"output_text"},"sequence_number":8,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"annotations":[],"text":"Let me.\n<tool_call>\n<function=run_q","type":"output_text"}],"id":"X","role":"assistant","status":"incomplete","type":"message"},"output_index":0,"sequence_number":9,"type":"response.output_item.done"}

event: response.incomplete
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":{"reason":"max_output_tokens"},"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"annotations":[],"text":"Let me.\n<tool_call>\n<function=run_q","type":"output_text"}],"id":"X","role":"assistant","status":"incomplete","type":"message"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"incomplete","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":{"input_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3}},"sequence_number":10,"type":"response.incomplete"}

data: [DONE]

)GOLDEN"},
        {"responses_length_in_reasoning", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"text":"","type":"reasoning_text"}],"id":"X","status":"in_progress","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"","type":"reasoning_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"weigh ","item_id":"X","output_index":0,"sequence_number":5,"type":"response.reasoning.delta"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"it ","item_id":"X","output_index":0,"sequence_number":6,"type":"response.reasoning.delta"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"ét","item_id":"X","output_index":0,"sequence_number":7,"type":"response.reasoning.delta"}

event: response.reasoning.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":8,"text":"weigh it ét","type":"response.reasoning.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"weigh it ét","type":"reasoning_text"},"sequence_number":9,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"text":"weigh it ét","type":"reasoning_text"}],"id":"X","status":"incomplete","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":10,"type":"response.output_item.done"}

event: response.incomplete
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":{"reason":"max_output_tokens"},"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"text":"weigh it ét","type":"reasoning_text"}],"id":"X","status":"incomplete","summary":[],"type":"reasoning"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"incomplete","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":{"input_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":3},"total_tokens":3}},"sequence_number":11,"type":"response.incomplete"}

data: [DONE]

)GOLDEN"},
        {"responses_think_call", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"text":"","type":"reasoning_text"}],"id":"X","status":"in_progress","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"","type":"reasoning_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"weigh it","item_id":"X","output_index":0,"sequence_number":5,"type":"response.reasoning.delta"}

event: response.reasoning.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":6,"text":"weigh it","type":"response.reasoning.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"weigh it","type":"reasoning_text"},"sequence_number":7,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":8,"type":"response.output_item.done"}

event: response.output_item.added
data: {"item":{"content":[{"annotations":[],"text":"","type":"output_text"}],"id":"X","role":"assistant","status":"in_progress","type":"message"},"output_index":1,"sequence_number":9,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":10,"type":"response.content_part.added"}

event: response.output_text.delta
data: {"content_index":0,"delta":"Running it.\n","item_id":"X","output_index":1,"sequence_number":11,"type":"response.output_text.delta"}

event: response.output_text.done
data: {"content_index":0,"item_id":"X","output_index":1,"sequence_number":12,"text":"Running it.\n","type":"response.output_text.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"Running it.\n","type":"output_text"},"sequence_number":13,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"annotations":[],"text":"Running it.\n","type":"output_text"}],"id":"X","role":"assistant","status":"completed","type":"message"},"output_index":1,"sequence_number":14,"type":"response.output_item.done"}

event: response.output_item.added
data: {"item":{"arguments":"","call_id":"call_0","id":"X","name":"run_query","status":"in_progress","type":"function_call"},"output_index":2,"sequence_number":15,"type":"response.output_item.added"}

event: response.function_call_arguments.delta
data: {"delta":"{\"sql\":\"SELECT 1 WHERE a < 2\"}","item_id":"X","output_index":2,"sequence_number":16,"type":"response.function_call_arguments.delta"}

event: response.function_call_arguments.done
data: {"arguments":"{\"sql\":\"SELECT 1 WHERE a < 2\"}","item_id":"X","output_index":2,"sequence_number":17,"type":"response.function_call_arguments.done"}

event: response.output_item.done
data: {"item":{"arguments":"{\"sql\":\"SELECT 1 WHERE a < 2\"}","call_id":"call_0","id":"X","name":"run_query","status":"completed","type":"function_call"},"output_index":2,"sequence_number":18,"type":"response.output_item.done"}

event: response.completed
data: {"response":{"background":false,"completed_at":0,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},{"content":[{"annotations":[],"text":"Running it.\n","type":"output_text"}],"id":"X","role":"assistant","status":"completed","type":"message"},{"arguments":"{\"sql\":\"SELECT 1 WHERE a < 2\"}","call_id":"call_0","id":"X","name":"run_query","status":"completed","type":"function_call"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"completed","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":{"input_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens":7,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":7}},"sequence_number":19,"type":"response.completed"}

data: [DONE]

)GOLDEN"},
        {"responses_think_call_notools", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"text":"","type":"reasoning_text"}],"id":"X","status":"in_progress","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"","type":"reasoning_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"weigh it","item_id":"X","output_index":0,"sequence_number":5,"type":"response.reasoning.delta"}

event: response.reasoning.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":6,"text":"weigh it","type":"response.reasoning.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"weigh it","type":"reasoning_text"},"sequence_number":7,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":8,"type":"response.output_item.done"}

event: response.output_item.added
data: {"item":{"content":[{"annotations":[],"text":"","type":"output_text"}],"id":"X","role":"assistant","status":"in_progress","type":"message"},"output_index":1,"sequence_number":9,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":10,"type":"response.content_part.added"}

event: response.output_text.delta
data: {"content_index":0,"delta":"Running it.\n","item_id":"X","output_index":1,"sequence_number":11,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"<tool_call>","item_id":"X","output_index":1,"sequence_number":12,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n","item_id":"X","output_index":1,"sequence_number":13,"type":"response.output_text.delta"}

event: response.output_text.delta
data: {"content_index":0,"delta":"</tool_call>","item_id":"X","output_index":1,"sequence_number":14,"type":"response.output_text.delta"}

event: response.output_text.done
data: {"content_index":0,"item_id":"X","output_index":1,"sequence_number":15,"text":"Running it.\n<tool_call>\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>","type":"response.output_text.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"Running it.\n<tool_call>\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>","type":"output_text"},"sequence_number":16,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"annotations":[],"text":"Running it.\n<tool_call>\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>","type":"output_text"}],"id":"X","role":"assistant","status":"completed","type":"message"},"output_index":1,"sequence_number":17,"type":"response.output_item.done"}

event: response.completed
data: {"response":{"background":false,"completed_at":0,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},{"content":[{"annotations":[],"text":"Running it.\n<tool_call>\n<function=run_query>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>","type":"output_text"}],"id":"X","role":"assistant","status":"completed","type":"message"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"completed","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":{"input_tokens":0,"input_tokens_details":{"cached_tokens":0},"output_tokens":7,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":7}},"sequence_number":18,"type":"response.completed"}

data: [DONE]

)GOLDEN"},
        {"responses_throws_mid_answer", R"GOLDEN(event: response.created
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":1,"type":"response.created"}

event: response.in_progress
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":null,"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"in_progress","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":2,"type":"response.in_progress"}

event: response.output_item.added
data: {"item":{"content":[{"text":"","type":"reasoning_text"}],"id":"X","status":"in_progress","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":3,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"","type":"reasoning_text"},"sequence_number":4,"type":"response.content_part.added"}

event: response.reasoning.delta
data: {"content_index":0,"delta":"weigh it","item_id":"X","output_index":0,"sequence_number":5,"type":"response.reasoning.delta"}

event: response.reasoning.done
data: {"content_index":0,"item_id":"X","output_index":0,"sequence_number":6,"text":"weigh it","type":"response.reasoning.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":0,"part":{"text":"weigh it","type":"reasoning_text"},"sequence_number":7,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},"output_index":0,"sequence_number":8,"type":"response.output_item.done"}

event: response.output_item.added
data: {"item":{"content":[{"annotations":[],"text":"","type":"output_text"}],"id":"X","role":"assistant","status":"in_progress","type":"message"},"output_index":1,"sequence_number":9,"type":"response.output_item.added"}

event: response.content_part.added
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"","type":"output_text"},"sequence_number":10,"type":"response.content_part.added"}

event: response.output_text.delta
data: {"content_index":0,"delta":"half an ans","item_id":"X","output_index":1,"sequence_number":11,"type":"response.output_text.delta"}

event: response.output_text.done
data: {"content_index":0,"item_id":"X","output_index":1,"sequence_number":12,"text":"half an ans","type":"response.output_text.done"}

event: response.content_part.done
data: {"content_index":0,"item_id":"X","output_index":1,"part":{"annotations":[],"text":"half an ans","type":"output_text"},"sequence_number":13,"type":"response.content_part.done"}

event: response.output_item.done
data: {"item":{"content":[{"annotations":[],"text":"half an ans","type":"output_text"}],"id":"X","role":"assistant","status":"incomplete","type":"message"},"output_index":1,"sequence_number":14,"type":"response.output_item.done"}

event: error
data: {"error":{"code":"internal_error","message":"backend exploded","param":null,"type":"server_error"},"sequence_number":15,"type":"error"}

event: response.failed
data: {"response":{"background":false,"completed_at":null,"created_at":0,"error":{"code":"internal_error","message":"backend exploded"},"frequency_penalty":0.0,"id":"X","incomplete_details":null,"instructions":null,"max_output_tokens":64,"max_tool_calls":null,"metadata":null,"model":"stub","object":"response","output":[{"content":[{"text":"weigh it","type":"reasoning_text"}],"id":"X","status":"completed","summary":[],"type":"reasoning"},{"content":[{"annotations":[],"text":"half an ans","type":"output_text"}],"id":"X","role":"assistant","status":"incomplete","type":"message"}],"parallel_tool_calls":true,"presence_penalty":0.0,"previous_response_id":null,"prompt_cache_key":null,"reasoning":null,"safety_identifier":null,"service_tier":"default","status":"failed","store":true,"temperature":0.699999988079071,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[{"description":null,"name":"run_query","parameters":{"properties":{"sql":{"type":"string"}},"type":"object"},"strict":null,"type":"function"}],"top_logprobs":0,"top_p":0.800000011920929,"truncation":"disabled","usage":null},"sequence_number":16,"type":"response.failed"}

data: [DONE]

)GOLDEN"},
    };
    return g;
}
