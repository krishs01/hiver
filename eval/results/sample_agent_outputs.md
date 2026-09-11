# Sample Agent Outputs

These are raw examples of the agent processing real customer messages.
Included as evidence artifacts for the evaluation.

---

## Example 1: Standard Troubleshooting (AUTO-HANDLE)

**Customer**: "@AppleSupport My iPhone keeps restarting on its own every few minutes after updating to the latest iOS"

**Agent Output**:
```
Intent:     SOFTWARE_UPDATE (confidence: 0.92)
Decision:   AUTO-HANDLE
Reply:      "We're sorry to hear about this! Try a force restart first: 
             press and quickly release Volume Up, then Volume Down, then 
             hold the Side button until you see the Apple logo. If the 
             restarts continue, let us know in a DM and we'll look into 
             it further with you."
```

**Retrieved similar cases**: 3/5 were iOS update restart issues; brand historically suggests force restart → DM for further diagnosis.

---

## Example 2: Account Security (ESCALATE)

**Customer**: "@AppleSupport Someone changed my Apple ID password and I can't get back in. I think my account was hacked!"

**Agent Output**:
```
Intent:     ACCOUNT_ACCESS (confidence: 0.95)
Decision:   ESCALATE
Reason:     Potential account security issue
Reply:      "We understand how concerning this must be. Your account 
             security is our top priority. Please visit iforgot.apple.com 
             immediately, and send us a DM so we can help secure your 
             account right away."
```

**Escalation signals**: `account_security` rule triggered ("hacked", "changed", "password").

---

## Example 3: Multi-Intent Message (Partial Failure)

**Customer**: "@AppleSupport My battery drains in 2 hours AND WiFi keeps disconnecting since the update"

**Agent Output**:
```
Intent:     BATTERY_POWER (confidence: 0.70)  ← missed CONNECTIVITY
Decision:   AUTO-HANDLE
Reply:      "That battery drain doesn't sound right. Let's try a few 
             things: go to Settings > Battery to check which apps are 
             using the most power. Also try Settings > General > Reset 
             > Reset Network Settings for the WiFi issue. DM us if 
             these steps don't help!"
```

**Note**: Classified as BATTERY_POWER (primary intent) but the reply actually addresses both issues thanks to RAG retrieving similar compound complaints. This is a case where the reply is better than the classification suggests.

---

## Example 4: Vague Message (Known Weakness)

**Customer**: "@AppleSupport help it's not working"

**Agent Output**:
```
Intent:     GENERAL_INQUIRY (confidence: 0.45)
Decision:   AUTO-HANDLE
Reply:      "We'd like to help! Could you let us know a bit more about 
             what's happening? Which device are you using, and what 
             exactly isn't working? The more details, the faster we 
             can assist."
```

**Note**: Low confidence correctly reflects ambiguity. The reply asks for clarification, which matches Apple's actual pattern for vague messages.

---

## Example 5: Frustrated Customer (Correct Escalation)

**Customer**: "@AppleSupport This is the 4th time I've contacted you about the same issue. Nobody has been able to fix it. I'm done with Apple products."

**Agent Output**:
```
Intent:     FEEDBACK_COMPLAINT (confidence: 0.88)
Decision:   ESCALATE
Reason:     Customer indicates repeated unresolved contact; 
             Customer showing extreme frustration
Reply:      "We sincerely apologize for the frustrating experience. 
             We understand this has been an ongoing issue and we want 
             to make it right. A senior advisor will follow up with 
             you directly. Please DM us your case details."
```

**Escalation signals**: `repeated_contact` ("4th time", "same issue") + `extreme_frustration` ("done with Apple").
