class OpenTTDLEBridgeInfo extends AIInfo {
    function GetAuthor() { return "OpenTTD-LE"; }
    function GetName() { return "OpenTTDLEBridge"; }
    function GetShortName() { return "OTLE"; }
    function GetDescription() { return "Bridge AI for OpenTTD-LE macro-action experiments."; }
    function GetVersion() { return 1; }
    function GetDate() { return "2026-05-14"; }
    function CreateInstance() { return "OpenTTDLEBridge"; }
    function GetAPIVersion() { return "1.0"; }
}

RegisterAI(OpenTTDLEBridgeInfo());
