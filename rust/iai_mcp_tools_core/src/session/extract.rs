//! Rule-based (non-LLM) classification of a raw tool-hook payload into a
//! `SessionEvent`. Ordered pattern match, most-specific first, so an error
//! or a marker-tagged prompt is never mis-bucketed as a routine
//! tool-command event.

use serde::Deserialize;

use super::{data_hash, rfc3339_now, SessionEvent};

/// All fields optional/defaulted so malformed or partial hook JSON never
/// fails to deserialize.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HookInput {
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub is_error: Option<bool>,
    #[serde(default)]
    pub tool_response: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub source_hook: Option<String>,
}

const PRIORITY_CRITICAL: u8 = 0;
const PRIORITY_SIGNAL: u8 = 1;
const PRIORITY_ROUTINE: u8 = 2;

const TEXT_MARKERS: &[(&str, &str)] = &[
    ("decision:", "decision"),
    ("rule:", "rule"),
    ("intent:", "intent"),
    ("goal:", "goal"),
    ("role:", "role"),
];

fn build_event(
    input: &HookInput,
    category: &str,
    event_type: &str,
    data: String,
    priority: u8,
) -> SessionEvent {
    SessionEvent {
        seq: 0,
        session_id: input.session_id.clone().unwrap_or_default(),
        ts: rfc3339_now(),
        category: category.to_string(),
        event_type: event_type.to_string(),
        data_hash: data_hash(&data),
        data,
        priority,
        source_hook: input.source_hook.clone().unwrap_or_default(),
    }
}

/// Matches a leading `"<marker>:"` prefix (case-insensitive on the marker
/// only) and returns the mapped category plus the trimmed remainder.
fn marker_category(prompt: &str) -> Option<(&'static str, &str)> {
    let trimmed = prompt.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    for (marker, category) in TEXT_MARKERS {
        if lower.starts_with(marker) {
            return Some((category, trimmed[marker.len()..].trim_start()));
        }
    }
    None
}

fn tool_response_is_error(resp: &str) -> bool {
    let lower = resp.to_ascii_lowercase();
    lower.contains("error") || lower.contains("exception") || lower.contains("traceback")
}

fn is_git_command(command: &str) -> bool {
    command.split_whitespace().any(|tok| tok == "git")
}

fn is_cwd_command(command: &str) -> bool {
    command.split_whitespace().next() == Some("cd")
        || command.split_whitespace().any(|tok| tok == "pwd")
}

fn is_env_command(command: &str) -> bool {
    command
        .split_whitespace()
        .any(|tok| matches!(tok, "export" | "unset" | "printenv"))
}

fn first_non_empty(candidates: &[Option<&str>]) -> String {
    candidates
        .iter()
        .filter_map(|c| *c)
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Maps a hook event to one of the 14 load-bearing categories (file, task,
/// rule, decision, cwd, error, env, git, subagent, intent, goal, skill,
/// role, user-prompt), most-specific pattern first. Returns `None` on an
/// unmapped tool event or on null/empty/malformed input -- never panics.
pub fn classify(input: &HookInput) -> Option<SessionEvent> {
    if input.is_error == Some(true) {
        let data = first_non_empty(&[
            input.tool_response.as_deref(),
            input.content.as_deref(),
            input.command.as_deref(),
        ]);
        return Some(build_event(
            input,
            "error",
            "tool_error",
            data,
            PRIORITY_CRITICAL,
        ));
    }
    if let Some(resp) = input.tool_response.as_deref() {
        if tool_response_is_error(resp) {
            return Some(build_event(
                input,
                "error",
                "tool_error",
                resp.to_string(),
                PRIORITY_CRITICAL,
            ));
        }
    }

    if let Some(prompt) = input.prompt.as_deref() {
        if let Some((category, rest)) = marker_category(prompt) {
            return Some(build_event(
                input,
                category,
                "text_marker",
                rest.to_string(),
                PRIORITY_SIGNAL,
            ));
        }
    }
    if input.source_hook.as_deref() == Some("UserPromptSubmit") {
        let data = input.prompt.clone().unwrap_or_default();
        return Some(build_event(
            input,
            "user-prompt",
            "user_prompt_submit",
            data,
            PRIORITY_CRITICAL,
        ));
    }

    let tool = input.tool_name.as_deref()?;
    match tool {
        "Edit" | "Write" | "MultiEdit" => {
            let file_path = input.file_path.as_deref()?;
            if file_path.is_empty() {
                return None;
            }
            Some(build_event(
                input,
                "file",
                tool,
                file_path.to_string(),
                PRIORITY_ROUTINE,
            ))
        }
        "TodoWrite" => Some(build_event(
            input,
            "task",
            "todo_write",
            input.content.clone().unwrap_or_default(),
            PRIORITY_ROUTINE,
        )),
        "Task" => Some(build_event(
            input,
            "subagent",
            "task_spawn",
            input.content.clone().unwrap_or_default(),
            PRIORITY_SIGNAL,
        )),
        "Skill" => Some(build_event(
            input,
            "skill",
            "skill_invoke",
            input.content.clone().unwrap_or_default(),
            PRIORITY_ROUTINE,
        )),
        "Bash" => {
            let command = input.command.as_deref().unwrap_or("");
            if command.is_empty() {
                return None;
            }
            if is_git_command(command) {
                Some(build_event(
                    input,
                    "git",
                    "git_command",
                    command.to_string(),
                    PRIORITY_ROUTINE,
                ))
            } else if is_cwd_command(command) {
                Some(build_event(
                    input,
                    "cwd",
                    "cwd_change",
                    command.to_string(),
                    PRIORITY_ROUTINE,
                ))
            } else if is_env_command(command) {
                Some(build_event(
                    input,
                    "env",
                    "env_command",
                    command.to_string(),
                    PRIORITY_ROUTINE,
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> HookInput {
        HookInput {
            session_id: Some("sess-x".to_string()),
            source_hook: Some("PostToolUse".to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn file_event_maps_edit_write_multiedit() {
        for tool in ["Edit", "Write", "MultiEdit"] {
            let input = HookInput {
                tool_name: Some(tool.to_string()),
                file_path: Some("src/x.rs".to_string()),
                ..base()
            };
            let event = classify(&input).unwrap();
            assert_eq!(event.category, "file");
        }
    }

    #[test]
    fn task_event_maps_todo_write() {
        let input = HookInput {
            tool_name: Some("TodoWrite".to_string()),
            content: Some("add tests".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "task");
        assert_eq!(event.data, "add tests");
    }

    #[test]
    fn git_event_maps_bash_git_command() {
        let input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("git status".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "git");
    }

    #[test]
    fn cwd_event_maps_bash_cd_command() {
        let input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("cd /tmp".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "cwd");
    }

    #[test]
    fn subagent_event_maps_task_tool() {
        let input = HookInput {
            tool_name: Some("Task".to_string()),
            content: Some("spawn reviewer".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "subagent");
    }

    #[test]
    fn skill_event_maps_skill_tool() {
        let input = HookInput {
            tool_name: Some("Skill".to_string()),
            content: Some("i-have-adhd".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "skill");
    }

    #[test]
    fn error_event_maps_is_error_flag() {
        let input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("pytest".to_string()),
            is_error: Some(true),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "error");
    }

    #[test]
    fn error_event_maps_tool_response_carrying_an_error() {
        let input = HookInput {
            tool_name: Some("Bash".to_string()),
            tool_response: Some("Traceback (most recent call last): ValueError".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "error");
    }

    #[test]
    fn env_event_maps_bash_export_command() {
        let input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("export FOO=bar".to_string()),
            ..base()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "env");
    }

    #[test]
    fn user_prompt_event_preserves_hyphenated_category() {
        let input = HookInput {
            source_hook: Some("UserPromptSubmit".to_string()),
            prompt: Some("what does this do".to_string()),
            session_id: Some("sess-x".to_string()),
            ..Default::default()
        };
        let event = classify(&input).unwrap();
        assert_eq!(event.category, "user-prompt");
    }

    #[test]
    fn decision_rule_intent_goal_role_events_map_from_text_markers() {
        let cases = [
            ("decision: ship v2", "decision", "ship v2"),
            ("rule: no force push", "rule", "no force push"),
            ("intent: refactor auth", "intent", "refactor auth"),
            ("goal: ship by friday", "goal", "ship by friday"),
            ("role: reviewer only", "role", "reviewer only"),
        ];
        for (prompt, category, rest) in cases {
            let input = HookInput {
                prompt: Some(prompt.to_string()),
                session_id: Some("sess-x".to_string()),
                ..Default::default()
            };
            let event = classify(&input).unwrap();
            assert_eq!(event.category, category);
            assert_eq!(event.data, rest);
        }
    }

    #[test]
    fn most_specific_pattern_wins_error_beats_command_rules() {
        let git_input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("git commit -m fail".to_string()),
            is_error: Some(true),
            ..base()
        };
        assert_eq!(classify(&git_input).unwrap().category, "error");

        let cwd_input = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some("cd /nonexistent".to_string()),
            is_error: Some(true),
            ..base()
        };
        assert_eq!(classify(&cwd_input).unwrap().category, "error");
    }

    #[test]
    fn classify_returns_none_on_null_empty_or_garbage_input_without_panicking() {
        assert!(classify(&HookInput::default()).is_none());

        let unmapped_tool = HookInput {
            tool_name: Some("Read".to_string()),
            ..base()
        };
        assert!(classify(&unmapped_tool).is_none());

        let empty_bash = HookInput {
            tool_name: Some("Bash".to_string()),
            command: Some(String::new()),
            ..base()
        };
        assert!(classify(&empty_bash).is_none());

        let empty_file_path = HookInput {
            tool_name: Some("Edit".to_string()),
            file_path: Some(String::new()),
            ..base()
        };
        assert!(classify(&empty_file_path).is_none());
    }
}
