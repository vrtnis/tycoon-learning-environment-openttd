class OpenTTDLECompanyInfo extends AIInfo {
    function GetAuthor() { return "OpenTTD-LE"; }
    function GetName() { return "OpenTTDLECompany"; }
    function GetShortName() { return "OTLC"; }
    function GetDescription() { return "Passive company holder for OpenTTD-LE GameScript control."; }
    function GetVersion() { return 1; }
    function GetDate() { return "2026-05-14"; }
    function CreateInstance() { return "OpenTTDLECompany"; }
    function GetAPIVersion() { return "1.0"; }
}

RegisterAI(OpenTTDLECompanyInfo());
